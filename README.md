# RAG Based SKU Classification System

## Overview
This project is an AI-powered SKU classification system for Thermo Fisher, designed to automate and enhance the process of classifying SKUs (Stock Keeping Units) using both traditional ML (TF-IDF) and modern RAG (Retrieval-Augmented Generation) techniques. It features a Streamlit web interface for single and bulk classification, feedback collection, analytics, and Microsoft Teams integration for notifications.

## Features
- **RAG Based SKU Classification:** Classify individual SKUs using RAG.
- **Bulk Classification:** Upload CSV/Excel files for batch processing of SKUs.
- **Feedback System:** Users can provide feedback (like/dislike) and corrections, which are saved and can be used to retrain the model.
- **Analytics Dashboard:** View feedback statistics and model performance.
- **Microsoft Teams Integration:** Sends notifications for logins and feedback to a Teams channel.
- **Embeddings Store:** Uses FAISS or sklearn for fast similarity search over SKU embeddings.

## File Structure
- `main.py`: Streamlit app, UI logic, authentication, feedback, analytics, and tab management.
- `rag_system.py`: RAG embedding index builder, search functions, and backend logic for similarity search (FAISS/sklearn).
- `teams_config.py`: Microsoft Teams webhook configuration and notification functions.
- `emb_store/`: Stores embeddings, metadata, and index files for fast retrieval.
- `feedback_data/`: Stores user feedback in CSV/JSON format.
- `user_credentials/`: Stores user login information.
- `requirements.txt`: Python dependencies.
- `Business_Rule.xlsx`, `Training_Set.xlsx`, `reference_file_hierechy.xlsx`: Data files for rules, training, and hierarchy.

## How It Works
### 1. RAG Embedding Index (`rag_system.py`)
- Uses `sentence-transformers` to encode SKU data into embeddings.
- Stores embeddings and metadata in `emb_store/`.
- Supports FAISS (if available) for fast similarity search, otherwise falls back to sklearn's NearestNeighbors.
- Provides functions:
  - `build_or_load_index(df, force_rebuild, model_name)`: Builds or loads the embedding index.
  - `search_top_k(llm_idx, query, top_k)`: Searches for top-k similar SKUs for a query.
  - `bulk_search_top_k(llm_idx, queries, top_k)`: Bulk search for multiple queries.

### 2. Teams Integration (`teams_config.py`)
- Configure your Teams webhook URL in `TEAMS_WEBHOOK_URL`.
- Functions:
  - `send_feedback_notification(...)`: Sends feedback (like/dislike) notifications to Teams, including user info, prediction, corrections, and comments.
  - `send_daily_summary_notification()`: Sends a daily summary of user activity and feedback to Teams.

### 3. Streamlit App (`main.py`)
- **Authentication:** User login with email, credentials saved locally.
- **Tabs:**
  - RAG Based Prediction: Single SKU classification.
  - Bulk Classification: Upload and process files for batch SKU classification.
  - Analytics Dashboard: View feedback and model stats.
- **Feedback:** Users can like/dislike predictions and provide corrections, which are saved and sent to Teams.
- **Session State:** Persists tab selection and user state for smooth navigation.

## Setup Instructions
1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure Teams Webhook:**
   - Go to your Teams channel > Connectors > Incoming Webhook > Configure.
   - Copy the webhook URL and set it in `teams_config.py` as `TEAMS_WEBHOOK_URL`.
3. **Prepare Data Files:**
   - Place `Business_Rule.xlsx`, `Training_Set.xlsx`, and `reference_file_hierechy.xlsx` in the project root.
4. **Run the App:**
   ```bash
   streamlit run main.py
   ```

## Feedback & Retraining
- User feedback is stored in `feedback_data/`.
- Admins can retrain the model using feedback data via the Analytics Dashboard.

## Embedding Store
- Embeddings and index files are stored in `emb_store/`.
- Supports both FAISS and sklearn for similarity search.

## Microsoft Teams Integration
- All login and feedback events are sent to the configured Teams channel.
- Daily summary notifications can be enabled.

## Requirements
- Python 3.8+
- `streamlit`, `pandas`, `numpy`, `sentence-transformers`, `faiss-cpu` (optional), `scikit-learn`, `requests`, etc.

## Security & Privacy
- User credentials are stored locally in `user_credentials/`.
- Feedback data is stored in `feedback_data/`.
- Teams notifications use the configured webhook URL.

## License
This project is for internal use at Thermo Fisher. Please contact the project owner for licensing details.

## Contact
For support or questions, contact the project maintainer or post in the configured Teams channel.
