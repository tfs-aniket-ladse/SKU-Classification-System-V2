# rag_system.py
from __future__ import annotations
import os
import json
import pickle
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

# Embeddings
from sentence_transformers import SentenceTransformer

# Choose backend: FAISS if available, otherwise sklearn KNN
USE_FAISS = False
try:
    import faiss  # type: ignore
    USE_FAISS = True
except Exception:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    USE_FAISS = False

STORE_DIR = "emb_store"
EMB_FILE = os.path.join(STORE_DIR, "embeddings.npy")
META_FILE = os.path.join(STORE_DIR, "meta.csv")
IDX_FILE  = os.path.join(STORE_DIR, "nn.pkl")          # sklearn index
FAISS_IDX = os.path.join(STORE_DIR, "faiss.index")     # faiss index
CFG_FILE  = os.path.join(STORE_DIR, "config.json")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Helpers ----------

def _ensure_store() -> None:
    if not os.path.exists(STORE_DIR):
        os.makedirs(STORE_DIR)

def _combine_text(df: pd.DataFrame) -> pd.Series:
    # Same idea as TF-IDF path
    return (df["sku number"].astype(str) + " " + df["sku name"].astype(str)).str.lower()

def _normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Only keep columns needed to render predictions
    keep = [
        "sku number", "sku name",
        "product line code", "product line name",
        "cmr product line", "sub platform"
    ]
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()

# ---------- Indexing ----------

def build_or_load_index(
    df: pd.DataFrame,
    force_rebuild: bool = False,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - model: SentenceTransformer
      - meta:  pd.DataFrame  (N rows, metadata aligned to embeddings row-by-row)
      - emb:   np.ndarray    (N x D, float32, normalized)
      - encode(texts)->np.ndarray
      - search(q_emb, top_k)->(indices, sims)
      - rebuild()->None
    """
    _ensure_store()

    # detect whether we must rebuild
    cfg = {}
    if os.path.exists(CFG_FILE):
        with open(CFG_FILE, "r") as f:
            cfg = json.load(f)

    need_build = (
        force_rebuild
        or not os.path.exists(EMB_FILE)
        or not os.path.exists(META_FILE)
        or cfg.get("model_name") != model_name
    )

    # Always instantiate model (fast after first load)
    model = SentenceTransformer(model_name)

    if need_build:
        texts = _combine_text(df).tolist()
        emb = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)

        meta = _normalize_rows(df)
        np.save(EMB_FILE, emb)
        meta.to_csv(META_FILE, index=False)

        cfg = {"model_name": model_name, "rows": int(emb.shape[0]), "dim": int(emb.shape[1]), "backend": "faiss" if USE_FAISS else "sklearn"}
        with open(CFG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)

        # build index
        if USE_FAISS:
            index = faiss.IndexFlatIP(emb.shape[1])  # cosine via dot on normalized vectors
            index.add(emb)
            faiss.write_index(index, FAISS_IDX)
        else:
            nn = NearestNeighbors(metric="cosine")
            nn.fit(emb)
            with open(IDX_FILE, "wb") as f:
                pickle.dump(nn, f)

    emb = np.load(EMB_FILE)
    meta = pd.read_csv(META_FILE)

    # Prepare search + encode closures
    if USE_FAISS and os.path.exists(FAISS_IDX):
        index = faiss.read_index(FAISS_IDX)

        def _search(q_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
            D, I = index.search(q_emb.astype(np.float32), top_k)
            # D are dot-products in [-1,1] (since normalized). Convert to [0,1] similarity if desired.
            sims = (D + 1.0) / 2.0
            return I, sims
    else:
        with open(IDX_FILE, "rb") as f:
            nn = pickle.load(f)

        def _search(q_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
            dists, inds = nn.kneighbors(q_emb, n_neighbors=top_k, return_distance=True)
            sims = 1.0 - dists  # cosine distance -> similarity
            return inds, sims

    def _encode(texts: List[str]) -> np.ndarray:
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _rebuild() -> None:
        build_or_load_index(df, force_rebuild=True, model_name=model_name)

    return {
        "model": model,
        "meta": meta,
        "emb": emb,
        "encode": _encode,
        "search": _search,
        "rebuild": _rebuild,
        "model_name": model_name,
        "cfg": cfg,
    }

def search_top_k(
    llm_idx: Dict[str, Any],
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Convenience single-query search that returns meta rows + similarity."""
    q_emb = llm_idx["encode"]([query])
    inds, sims = llm_idx["search"](q_emb, top_k=top_k)
    inds, sims = inds[0], sims[0]
    out = []
    for i, s in zip(inds, sims):
        row = llm_idx["meta"].iloc[int(i)].to_dict()
        row["similarity"] = float(s)
        out.append(row)
    return out

def bulk_search_top_k(
    llm_idx: Dict[str, Any],
    queries: List[str],
    top_k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized search for bulk:
      returns (indices, sims)
      - indices shape: [B, top_k]
      - sims    shape: [B, top_k]
    """
    q_emb = llm_idx["encode"](queries)
    inds, sims = llm_idx["search"](q_emb, top_k=top_k)
    return inds, sims