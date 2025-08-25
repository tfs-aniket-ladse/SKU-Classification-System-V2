from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import warnings
import os
from datetime import datetime

# Import LLM system
try:
    from rag_system import build_or_load_index, search_top_k, bulk_search_top_k
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    build_or_load_index = None
    search_top_k = None
    bulk_search_top_k = None

warnings.filterwarnings('ignore')
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

app = Flask(__name__)

# Import Teams configuration
try:
    from teams_config import TEAMS_WEBHOOK_URL
except ImportError:
    TEAMS_WEBHOOK_URL = ''

# Global variables for model data
df = None
vectorizer = None
tfidf_matrix = None
df_hierarchy = None
llm = None  # LLM index

def load_data():
    """Load and preprocess training data"""
    try:
        df = pd.read_excel('Training_Set.xlsx')
        df = df[(df['sku number'].notna()) & (df['sku number'] != '') & 
                (df['sku name'].notna()) & (df['sku name'] != '')]
        
        valid_cmr_product_lines = [
            'BEAService', 'BEAHardware', 'BEAOther', 'HardwareConsumables', 
            'SUTAutomation','2DBioProcessContainers', '3DBioProcessContainers', 
            'FillFinish', 'FlexibleOther','FluidTransferAssemblies', 
            'BioproductionContainments', 'BottleAssemblies',
            'ProductionCellCulture', 'RigidOther', 'SUDOther'
        ]
        df = df[df['cmr product line'].isin(valid_cmr_product_lines)]
        df = df.drop_duplicates(subset=['sku number', 'sku name'])
        
        # Apply volume-based CMR product line correction during data loading
        df['cmr product line'] = df.apply(lambda row: determine_correct_cmr_by_volume(
            row['sku name'], row['cmr product line']), axis=1)
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def load_reference_hierarchy():
    """Load reference hierarchy data"""
    try:
        return pd.read_excel('reference_file_hierechy.xlsx')
    except Exception as e:
        print(f"Error loading reference hierarchy: {str(e)}")
        return None

def create_similarity_index(df):
    """Create TF-IDF similarity index"""
    combined_text = (df['sku number'].astype(str) + " " + 
                    df['sku name'].astype(str)).str.lower()
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=3000,
        analyzer='char_wb',
        lowercase=True,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    return vectorizer, tfidf_matrix

def extract_volume_enhanced(sku_name):
    """Enhanced volume extraction"""
    if not sku_name:
        return None
    
    sku_name = str(sku_name).upper().strip()
    
    # Liter patterns
    liter_patterns = [
        r'(\d+(?:\.\d+)?)\s*L(?:\s|$|[^\w])',
        r'(\d+(?:\.\d+)?)\s*LITER',
        r'(\d+(?:\.\d+)?)\s*LITRE'
    ]
    
    for pattern in liter_patterns:
        match = re.search(pattern, sku_name)
        if match:
            return float(match.group(1))
    
    # Milliliter patterns
    ml_patterns = [
        r'(\d+(?:\.\d+)?)\s*M?ML(?:\s|$|[^\w])',
        r'(\d+(?:\.\d+)?)\s*MILLILITER',
        r'(\d+(?:\.\d+)?)\s*MILLILITRE'
    ]
    
    for pattern in ml_patterns:
        match = re.search(pattern, sku_name)
        if match:
            return float(match.group(1)) / 1000
    
    return None

def determine_correct_cmr_by_volume(sku_name, original_cmr_line):
    """Determine correct CMR product line based on 50L volume rule"""
    if original_cmr_line not in ['2DBioProcessContainers', '3DBioProcessContainers']:
        return original_cmr_line
    
    volume_l = extract_volume_enhanced(sku_name)
    if volume_l is None:
        return original_cmr_line
    
    return '2DBioProcessContainers' if volume_l <= 50 else '3DBioProcessContainers'

def create_2d_to_3d_mapping():
    """Create mapping from 2D product line codes to appropriate 3D codes"""
    return {
        '2JE': '2MH',  # GENERAL 2D -> PRODUCTAINER BPC
        '2JC': '2MH',  # LABTAINER -> PRODUCTAINER BPC
        '2PQ': '2PS',  # 2DBioProcessContainers Tieout -> 3DBioProcessContainers Tieout
        '2MD': '2MN',  # Map to 3D Manifold
        '2JD': '2MH',  # Map to PRODUCTAINER BPC
        '0CF': '0D8',  # 2D SINGLE -> SINGLE
        '2MB': '2MH',  # Map to PRODUCTAINER BPC
        '2MF': '2MJ',  # 2D TANK LINER -> 3D TANK LINERS
        '0D0': '2MN',  # MANIFOLD -> 3D MANIFOLD
        'Z3U': '2MH',  # Map to PRODUCTAINER BPC
        'Z6R': '0D8',  # 2D SINGLE -> SINGLE
        '0CZ': '0D8',  # 2D SINGLE -> SINGLE
        'Z3R': '0D8',  # 2D SINGLE -> SINGLE
        'Z2K': '2MN',  # MANIFOLD -> 3D MANIFOLD
        'Z37': '0D8'   # 2D SINGLE -> SINGLE
    }

def get_3d_to_2d_mapping():
    """3D to 2D product line code mapping"""
    return {
        '2MH': '2JE', '2MJ': '2MF', '2MO': '2JE', '2PS': '2JE', '2MN': '2MD',
        'Z2H': '2JE', '2ML': '2JE', 'Z39': '2JE', 'Z6M': '2JE', '2MM': '2JE',
        '0D8': '2MD', '0EG': '2JE', '3D6': '2JE', '3WO': '2JE', 'Z3Q': '2JE',
        '262': '2JE', '2MG': '2JE'
    }

def get_product_line_name(product_line_code, is_2d=True):
    """Get appropriate product line name"""
    if is_2d:
        mapping = {
            '2JE': 'GENERAL 2D', '2JC': 'LABTAINER', '2JD': 'LABTAINER PRO',
            '2PQ': '2DBioProcessContainers Tieout', '2MD': '2D MANIFOLD', '0CF': '2D SINGLE',
            '2MB': '2D HARVESTAINER', '2MF': '2D TANK LINER', '0D0': '2D MANIFOLD',
            'Z3U': 'MANIFOLD', 'Z6R': '2D SINGLE', '0CZ': '2D MANIFOLD', 'Z3R': '2D SINGLE',
            'Z2K': 'MANIFOLD', 'Z37': 'FLEXIBLE CONSUMABLES 2D'
        }
        return mapping.get(product_line_code, 'GENERAL 2D')
    else:
        mapping = {
            '2MH': 'PRODUCTAINER BPC', '2MJ': '3D TANK LINERS', '2MO': '3D PRODUCTAINER',
            '2PS': '3D PRODUCTAINER', '2MN': '3D MANIFOLD', 'Z2H': '3D PRODUCTAINER',
            '2ML': '3D PRODUCTAINER', 'Z39': '3D PRODUCTAINER', 'Z6M': '3D PRODUCTAINER',
            '2MM': 'OTHER OUTER SUPPORT CONTAINERS', '0D8': '3D MANIFOLD', '0EG': '3D PRODUCTAINER',
            '3D6': '3D PRODUCTAINER', '3WO': '3D PRODUCTAINER', 'Z3Q': '3D PRODUCTAINER',
            '262': '3D PRODUCTAINER', '2MG': '3D PRODUCTAINER'
        }
        return mapping.get(product_line_code, 'PRODUCTAINER BPC')

def get_product_line_lv2(product_line_code, df_hierarchy):
    """Get Product Line - LV 2 from product line code"""
    if df_hierarchy is None:
        return 'N/A'
    
    try:
        match = df_hierarchy[df_hierarchy['PL Codes'] == product_line_code]
        if not match.empty:
            return match.iloc[0]['Product Line - LV 2']
        return 'N/A'
    except Exception:
        return 'N/A'

def adjust_product_line_for_volume(original_cmr, product_line_code, product_line_name, sku_name):
    """Adjust product line code and name based on volume-determined CMR classification"""
    sku_name_upper = str(sku_name).upper() if sku_name else ""
    
    if "SPIGOT NEEDLE" in sku_name_upper:
        return "2L0", "BE20 TANK FITTINGS", "BioproductionContainments"
    if "PILLOW" in sku_name_upper:
        return "2JE", "GENERAL 2D", "2DBioProcessContainers"
    
    # EVA rule: any number X any number with EVA
    if "EVA" in sku_name_upper and re.search(r'\d+\s*[X*]\s*\d+', sku_name_upper):
        # Check for manifold in EVA SKUs
        if "MANF" in sku_name_upper or "MANIFOLD" in sku_name_upper:
            volume_l = extract_volume_enhanced(sku_name)
            if volume_l is not None:
                if volume_l <= 50:
                    return "2MD", "2D MANIFOLD", "2DBioProcessContainers"
                else:
                    return "2MN", "3D MANIFOLD", "3DBioProcessContainers"
            else:
                return "2MD", "2D MANIFOLD", "2DBioProcessContainers"
        return "2JE", "GENERAL 2D", "2DBioProcessContainers"
    
    if "BETA BAG" in sku_name_upper or "NEEDLE" in sku_name_upper:
        return "2NK", "FF FILLING ASSEMBLIES", "FillFinish"
    
    if original_cmr not in ['2DBioProcessContainers', '3DBioProcessContainers']:
        return product_line_code, product_line_name, original_cmr

    correct_cmr = determine_correct_cmr_by_volume(sku_name, original_cmr)

    # Always map 2D code to 3D code if CMR is 3DBioProcessContainers
    if correct_cmr == '3DBioProcessContainers':
        mapped_code = create_2d_to_3d_mapping().get(product_line_code, product_line_code)
        mapped_name = get_product_line_name(mapped_code, is_2d=False)
        return mapped_code, mapped_name, correct_cmr
    # Always map 3D code to 2D code if CMR is 2DBioProcessContainers
    elif correct_cmr == '2DBioProcessContainers':
        mapped_code = get_3d_to_2d_mapping().get(product_line_code, product_line_code)
        mapped_name = get_product_line_name(mapped_code, is_2d=True)
        return mapped_code, mapped_name, correct_cmr
    # Fallback (should not hit)
    return product_line_code, product_line_name, correct_cmr

@lru_cache(maxsize=500)
def send_teams_notification(sku_number, sku_name, endpoint_type):
    """Send notification to Teams channel"""
    if not TEAMS_WEBHOOK_URL:
        return
    
    try:
        import requests
        message = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": "Single SKU Request",
            "themeColor": "0076D7",
            "sections": [{
                "activityTitle": f"Single SKU Request - {endpoint_type}",
                "activitySubtitle": f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "facts": [
                    {"name": "SKU Number", "value": sku_number or "N/A"},
                    {"name": "SKU Name", "value": sku_name or "N/A"},
                    {"name": "Endpoint", "value": endpoint_type}
                ]
            }]
        }
        requests.post(TEAMS_WEBHOOK_URL, json=message, timeout=5)
    except:
        pass

@lru_cache(maxsize=500)
def calculate_simple_similarity(s1, s2):
    """Cached similarity calculation"""
    if not s1 or not s2:
        return 0.0
    
    s1, s2 = s1.lower(), s2.lower()
    if s1 in s2 or s2 in s1:
        return 85.0
    
    set1, set2 = set(s1), set(s2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return (intersection / union) * 100 if union > 0 else 0.0

def check_non_inventory_item(sku_number, sku_name):
    """Check if SKU is a non-inventory item"""
    sku_upper = str(sku_number).upper() if sku_number else ""
    name_upper = str(sku_name).upper() if sku_name else ""
    
    non_inventory_keywords = ['$FREIGHT', 'QFREIGHT', 'QFEE', 'QEXPEDITE', 'QRESTOCK', '$MISC', 'DOSE AUDIT', 'FADJ']
    
    for keyword in non_inventory_keywords:
        if keyword in sku_upper or keyword in name_upper:
            return True
    return False

def get_fuzzy_predictions(df, sku_partial, name_partial, vectorizer, tfidf_matrix, top_k=3):
    """Get fuzzy predictions with volume-based 2D/3D mapping"""
    if not sku_partial.strip() and not name_partial.strip():
        return []
    
    # Check for non-inventory items first
    if check_non_inventory_item(sku_partial, name_partial):
        return [{
            'SKU Number': 'Non-Inventory Item',
            'SKU Name': 'This SKU has been identified as a Non-Inventory item',
            'Product Line Code': 'N/A',
            'CMR Product Line': 'N/A',
            'Product Line Name': 'N/A',
            'Product Line Lv2': 'N/A',
            'Business Unit': 'N/A',
            'SKU Score': 100.0,
            'Name Score': 100.0,
            'Confidence Score': 100.0
        }]
    
    query_text = f"{sku_partial} {name_partial}".lower()
    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    top_indices = np.argsort(similarities)[-top_k*3:][::-1]
    results = []
    seen_combinations = set()
    
    for idx in top_indices:
        if len(results) >= top_k or similarities[idx] < 0.1:
            continue
            
        row = df.iloc[idx]
        combination_id = f"{row['product line code']}|{row['cmr product line']}"
        
        if combination_id not in seen_combinations:
            seen_combinations.add(combination_id)
            
            # Apply volume-based mapping using input name
            adj_code, adj_name, correct_cmr = adjust_product_line_for_volume(
                row['cmr product line'], row['product line code'], 
                row['product line name'], name_partial
            )
            
            # Get Product Line - LV 2
            product_line_lv2 = get_product_line_lv2(adj_code, df_hierarchy)
            
            results.append({
                'SKU Number': row['sku number'],
                'SKU Name': row['sku name'],
                'Product Line Code': adj_code,
                'CMR Product Line': correct_cmr,
                'Product Line Name': adj_name,
                'Product Line Lv2': product_line_lv2,
                'Business Unit': row['sub platform'],
                'SKU Score': round(calculate_simple_similarity(sku_partial, str(row['sku number'])), 2),
                'Name Score': round(calculate_simple_similarity(name_partial, str(row['sku name'])), 2),
                'Confidence Score': round(similarities[idx] * 100, 2)
            })
    
    return results

def get_llm_predictions(df, sku_partial, name_partial, top_k=3):
    """
    Get LLM predictions using embedding-based search
    Returns same format as get_fuzzy_predictions
    """
    if not sku_partial.strip() and not name_partial.strip():
        return []
    
    # Check for non-inventory items first
    if check_non_inventory_item(sku_partial, name_partial):
        return [{
            'SKU Number': 'Non-Inventory Item',
            'SKU Name': 'This SKU has been identified as a Non-Inventory item',
            'Product Line Code': 'N/A',
            'CMR Product Line': 'N/A',
            'Product Line Name': 'N/A',
            'Product Line Lv2': 'N/A',
            'Business Unit': 'N/A',
            'SKU Score': 100.0,
            'Name Score': 100.0,
            'Confidence Score': 100.0
        }]

    query_text = f"{sku_partial} {name_partial}".strip().lower()
    if not query_text:
        return []

    if not RAG_AVAILABLE or llm is None:
        # Fallback to TF-IDF when LLM not available
        return get_fuzzy_predictions(df, sku_partial, name_partial, vectorizer, tfidf_matrix, top_k)

    # Overfetch 3x to diversify like TF-IDF path
    raw = search_top_k(llm, query_text, top_k=top_k * 3)

    results = []
    seen = set()

    for r in raw:
        if len(results) >= top_k or r.get("similarity", 0) < 0.10:
            continue

        combo = f"{r.get('product line code')}|{r.get('cmr product line')}"
        if combo in seen:
            continue
        seen.add(combo)

        # Apply volume-based mapping using INPUT name (same behavior as TF-IDF)
        adj_code, adj_name, correct_cmr = adjust_product_line_for_volume(
            r.get('cmr product line', ''),
            r.get('product line code', ''),
            r.get('product line name', ''),
            name_partial
        )

        product_line_lv2 = get_product_line_lv2(adj_code, df_hierarchy)

        results.append({
            'SKU Number': r.get('sku number', ''),
            'SKU Name': r.get('sku name', ''),
            'Product Line Code': adj_code,
            'CMR Product Line': correct_cmr,
            'Product Line Name': adj_name,
            'Product Line Lv2': product_line_lv2,
            'Business Unit': r.get('sub platform', ''),
            'SKU Score': round(calculate_simple_similarity(sku_partial, str(r.get('sku number', ''))), 2),
            'Name Score': round(calculate_simple_similarity(name_partial, str(r.get('sku name', ''))), 2),
            'Confidence Score': round(float(r.get("similarity", 0.0)) * 100.0, 2)
        })

    return results

def get_bulk_llm_predictions(input_df):
    """
    Bulk LLM predictions for multiple SKUs
    Returns results in the same format as TF-IDF bulk predictions
    """
    if llm is None:
        return []

    # Normalize inputs
    input_clean = input_df.copy()
    input_clean['sku number'] = input_clean['sku number'].fillna('').astype(str)
    input_clean['sku name'] = input_clean['sku name'].fillna('').astype(str)
    queries = (input_clean['sku number'] + ' ' + input_clean['sku name']).str.lower().tolist()

    # Bulk search
    inds, sims = bulk_search_top_k(llm, queries, top_k=1)

    results = []
    meta = llm["meta"]
    
    for i, (_, row) in enumerate(input_df.iterrows()):
        result_row = row.copy()
        
        if inds.shape[1] > 0 and sims[i, 0] >= 0.10:
            mrow = meta.iloc[int(inds[i, 0])]
            
            # Apply volume-based mapping using *input* name
            adj_code, adj_name, correct_cmr = adjust_product_line_for_volume(
                mrow['cmr product line'],
                mrow['product line code'],
                mrow['product line name'],
                row.get('sku name', '')
            )
            
            product_line_lv2 = get_product_line_lv2(adj_code, df_hierarchy)

            prefix = 'LLM Prediction 1: '
            result_row[f'{prefix}SKU Number'] = mrow['sku number']
            result_row[f'{prefix}SKU Name'] = mrow['sku name']
            result_row[f'{prefix}CMR Product Line'] = correct_cmr
            result_row[f'{prefix}Product Line Name'] = adj_name
            result_row[f'{prefix}Product Line Code'] = adj_code
            result_row[f'{prefix}Product Line - LV 2'] = product_line_lv2
            result_row[f'{prefix}Business Unit'] = mrow['sub platform']
            result_row[f'{prefix}Confidence Score'] = round(float(sims[i, 0]) * 100.0, 2)
        else:
            prefix = 'LLM Prediction 1: '
            for suffix in ['SKU Number', 'SKU Name', 'CMR Product Line',
                           'Product Line Name', 'Product Line Code', 'Business Unit']:
                result_row[f'{prefix}{suffix}'] = 'No Match Found'
            result_row[f'{prefix}Confidence Score'] = 0.0
        
        results.append(result_row)

    return pd.DataFrame(results)

@app.route('/classify_single', methods=['POST'])
def classify_single():
    """Single SKU classification endpoint with both TF-IDF and LLM predictions"""
    try:
        data = request.get_json()
        sku_number = data.get('sku_number', '')
        sku_name = data.get('sku_name', '')
        prediction_type = data.get('prediction_type', 'tfidf')  # 'tfidf', 'llm', or 'both'
        
        if not sku_number.strip() and not sku_name.strip():
            return jsonify({'error': 'Please provide at least one field (sku_number or sku_name)'}), 400
        
        result = {
            'status': 'success',
            'input': {
                'sku_number': sku_number,
                'sku_name': sku_name
            }
        }
        
        # TF-IDF prediction
        if prediction_type in ['tfidf', 'both']:
            tfidf_predictions = get_fuzzy_predictions(df, sku_number, sku_name, vectorizer, tfidf_matrix, top_k=1)
            result['tfidf_prediction'] = tfidf_predictions[0] if tfidf_predictions else None
        
        # LLM prediction
        if prediction_type in ['llm', 'both']:
            if RAG_AVAILABLE and llm is not None:
                llm_predictions = get_llm_predictions(df, sku_number, sku_name, top_k=1)
                result['llm_prediction'] = llm_predictions[0] if llm_predictions else None
            else:
                result['llm_prediction'] = None
                result['llm_status'] = 'LLM system not available, using TF-IDF fallback'
        
        # For backward compatibility, if no prediction_type specified, return TF-IDF as 'prediction'
        if prediction_type == 'tfidf' or (prediction_type not in ['llm', 'both']):
            result['prediction'] = result.get('tfidf_prediction')
        
        # Send Teams notification
        send_teams_notification(sku_number, sku_name, "Single SKU Classification")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify_bulk', methods=['POST'])
def classify_bulk():
    """Bulk SKU classification endpoint with both TF-IDF and LLM predictions"""
    try:
        data = request.get_json()
        skus = data.get('skus', [])
        prediction_type = data.get('prediction_type', 'tfidf')  # 'tfidf', 'llm', or 'both'
        
        if not skus:
            return jsonify({'error': 'Please provide SKUs list'}), 400
        
        # Convert to DataFrame for bulk processing
        input_df = pd.DataFrame(skus)
        
        # Ensure required columns exist
        if 'sku_number' not in input_df.columns:
            input_df['sku_number'] = ''
        if 'sku_name' not in input_df.columns:
            input_df['sku_name'] = ''
        
        # Rename columns to match internal format
        input_df = input_df.rename(columns={'sku_number': 'sku number', 'sku_name': 'sku name'})
        
        results = []
        
        if prediction_type in ['tfidf', 'both']:
            # TF-IDF bulk predictions
            for _, row in input_df.iterrows():
                sku_number = str(row.get('sku number', ''))
                sku_name = str(row.get('sku name', ''))
                
                tfidf_predictions = get_fuzzy_predictions(df, sku_number, sku_name, vectorizer, tfidf_matrix, top_k=1)
                
                result = {
                    'input': {
                        'sku_number': sku_number,
                        'sku_name': sku_name
                    },
                    'tfidf_predictions': tfidf_predictions
                }
                results.append(result)
        
        if prediction_type in ['llm', 'both']:
            # LLM bulk predictions
            if RAG_AVAILABLE and llm is not None:
                llm_results_df = get_bulk_llm_predictions(input_df)
                
                # Merge LLM results with existing results
                for i, (_, row) in enumerate(input_df.iterrows()):
                    if i < len(results):
                        # Extract LLM predictions from the result row
                        llm_predictions = []
                        llm_row = llm_results_df.iloc[i] if i < len(llm_results_df) else None
                        
                        if llm_row is not None:
                            prefix = 'LLM Prediction 1: '
                            if llm_row.get(f'{prefix}SKU Number', 'No Match Found') != 'No Match Found':
                                llm_prediction = {
                                    'SKU Number': llm_row.get(f'{prefix}SKU Number', ''),
                                    'SKU Name': llm_row.get(f'{prefix}SKU Name', ''),
                                    'Product Line Code': llm_row.get(f'{prefix}Product Line Code', ''),
                                    'CMR Product Line': llm_row.get(f'{prefix}CMR Product Line', ''),
                                    'Product Line Name': llm_row.get(f'{prefix}Product Line Name', ''),
                                    'Product Line Lv2': llm_row.get(f'{prefix}Product Line - LV 2', ''),
                                    'Business Unit': llm_row.get(f'{prefix}Business Unit', ''),
                                    'Confidence Score': llm_row.get(f'{prefix}Confidence Score', 0.0)
                                }
                                llm_predictions.append(llm_prediction)
                        
                        results[i]['llm_predictions'] = llm_predictions
                    else:
                        # Create new result entry if TF-IDF not requested
                        sku_number = str(row.get('sku number', ''))
                        sku_name = str(row.get('sku name', ''))
                        
                        llm_predictions = []
                        llm_row = llm_results_df.iloc[i] if i < len(llm_results_df) else None
                        
                        if llm_row is not None:
                            prefix = 'LLM Prediction 1: '
                            if llm_row.get(f'{prefix}SKU Number', 'No Match Found') != 'No Match Found':
                                llm_prediction = {
                                    'SKU Number': llm_row.get(f'{prefix}SKU Number', ''),
                                    'SKU Name': llm_row.get(f'{prefix}SKU Name', ''),
                                    'Product Line Code': llm_row.get(f'{prefix}Product Line Code', ''),
                                    'CMR Product Line': llm_row.get(f'{prefix}CMR Product Line', ''),
                                    'Product Line Name': llm_row.get(f'{prefix}Product Line Name', ''),
                                    'Product Line Lv2': llm_row.get(f'{prefix}Product Line - LV 2', ''),
                                    'Business Unit': llm_row.get(f'{prefix}Business Unit', ''),
                                    'Confidence Score': llm_row.get(f'{prefix}Confidence Score', 0.0)
                                }
                                llm_predictions.append(llm_prediction)
                        
                        result = {
                            'input': {
                                'sku_number': sku_number,
                                'sku_name': sku_name
                            },
                            'llm_predictions': llm_predictions
                        }
                        results.append(result)
            else:
                # Add empty LLM predictions if LLM not available
                for result in results:
                    result['llm_predictions'] = []
                    result['llm_status'] = 'LLM system not available'
        
        return jsonify({
            'status': 'success',
            'total_processed': len(results),
            'prediction_type': prediction_type,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify_single_llm', methods=['POST'])
def classify_single_llm():
    """Dedicated LLM single SKU classification endpoint"""
    try:
        data = request.get_json()
        sku_number = data.get('sku_number', '')
        sku_name = data.get('sku_name', '')
        
        if not sku_number.strip() and not sku_name.strip():
            return jsonify({'error': 'Please provide at least one field (sku_number or sku_name)'}), 400
        
        if not RAG_AVAILABLE or llm is None:
            return jsonify({'error': 'LLM system not available'}), 503
        
        llm_predictions = get_llm_predictions(df, sku_number, sku_name, top_k=1)
        
        # Send Teams notification
        send_teams_notification(sku_number, sku_name, "Single SKU LLM Prediction")
        
        return jsonify({
            'status': 'success',
            'input': {
                'sku_number': sku_number,
                'sku_name': sku_name
            },
            'predictions': llm_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify_bulk_llm', methods=['POST'])
def classify_bulk_llm():
    """Dedicated LLM bulk SKU classification endpoint"""
    try:
        data = request.get_json()
        skus = data.get('skus', [])
        
        if not skus:
            return jsonify({'error': 'Please provide SKUs list'}), 400
        
        if not RAG_AVAILABLE or llm is None:
            return jsonify({'error': 'LLM system not available'}), 503
        
        # Convert to DataFrame
        input_df = pd.DataFrame(skus)
        
        # Ensure required columns exist
        if 'sku_number' not in input_df.columns:
            input_df['sku_number'] = ''
        if 'sku_name' not in input_df.columns:
            input_df['sku_name'] = ''
        
        # Rename columns to match internal format
        input_df = input_df.rename(columns={'sku_number': 'sku number', 'sku_name': 'sku name'})
        
        # Get LLM predictions
        llm_results_df = get_bulk_llm_predictions(input_df)
        
        # Convert results to API format
        results = []
        for i, (_, row) in enumerate(input_df.iterrows()):
            sku_number = str(row.get('sku number', ''))
            sku_name = str(row.get('sku name', ''))
            
            # Extract LLM predictions from the result row
            llm_predictions = []
            llm_row = llm_results_df.iloc[i] if i < len(llm_results_df) else None
            
            if llm_row is not None:
                prefix = 'LLM Prediction 1: '
                if llm_row.get(f'{prefix}SKU Number', 'No Match Found') != 'No Match Found':
                    llm_prediction = {
                        'SKU Number': llm_row.get(f'{prefix}SKU Number', ''),
                        'SKU Name': llm_row.get(f'{prefix}SKU Name', ''),
                        'Product Line Code': llm_row.get(f'{prefix}Product Line Code', ''),
                        'CMR Product Line': llm_row.get(f'{prefix}CMR Product Line', ''),
                        'Product Line Name': llm_row.get(f'{prefix}Product Line Name', ''),
                        'Product Line Lv2': llm_row.get(f'{prefix}Product Line - LV 2', ''),
                        'Business Unit': llm_row.get(f'{prefix}Business Unit', ''),
                        'Confidence Score': llm_row.get(f'{prefix}Confidence Score', 0.0)
                    }
                    llm_predictions.append(llm_prediction)
            
            result = {
                'input': {
                    'sku_number': sku_number,
                    'sku_name': sku_name
                },
                'predictions': llm_predictions
            }
            results.append(result)
        
        return jsonify({
            'status': 'success',
            'total_processed': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': df is not None,
        'total_training_records': len(df) if df is not None else 0,
        'llm_available': RAG_AVAILABLE and llm is not None,
        'tfidf_available': vectorizer is not None and tfidf_matrix is not None
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    info = {
        'training_records': len(df) if df is not None else 0,
        'tfidf_available': vectorizer is not None and tfidf_matrix is not None,
        'llm_available': RAG_AVAILABLE and llm is not None,
        'hierarchy_data_loaded': df_hierarchy is not None
    }
    
    if df is not None:
        info['unique_product_lines'] = df['product line code'].nunique()
        info['unique_cmr_lines'] = df['cmr product line'].nunique()
        info['unique_business_units'] = df['sub platform'].nunique()
    
    return jsonify(info)

def initialize_model():
    """Initialize the model on startup"""
    global df, vectorizer, tfidf_matrix, df_hierarchy, llm
    
    print("Loading training data...")
    df = load_data()
    if df is None:
        raise Exception("Failed to load training data")
    
    print("Loading reference hierarchy...")
    df_hierarchy = load_reference_hierarchy()
    
    print("Creating TF-IDF similarity index...")
    vectorizer, tfidf_matrix = create_similarity_index(df)
    
    # Initialize LLM system if available
    if RAG_AVAILABLE and build_or_load_index is not None:
        try:
            print("Initializing LLM embeddings store...")
            llm = build_or_load_index(df, force_rebuild=False)
            print("LLM system initialized successfully")
        except Exception as e:
            print(f"Could not initialize LLM system: {str(e)}. Using TF-IDF only.")
            llm = None
    else:
        print("LLM system not available. Using TF-IDF only.")
        llm = None
    
    print(f"Model initialized successfully with {len(df)} training records")
    print(f"TF-IDF available: {vectorizer is not None}")
    print(f"LLM available: {llm is not None}")

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000)