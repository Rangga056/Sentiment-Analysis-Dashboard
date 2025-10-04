import streamlit as st
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.text_processing import DEFAULT_NORMALIZATION_DICT, DEFAULT_VETO_WORDS

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Data storage
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    
    if 'df_sentiment' not in st.session_state:
        st.session_state.df_sentiment = None
    
    if 'text_column' not in st.session_state:
        st.session_state.text_column = None
    
    # Preprocessing options
    if 'preprocessing_options' not in st.session_state:
        st.session_state.preprocessing_options = {
            'use_stemming': True,
            'use_stopwords': True,
            'min_text_length': 15
        }
    
    # Normalization dictionary
    if 'normalization_dict' not in st.session_state:
        st.session_state.normalization_dict = DEFAULT_NORMALIZATION_DICT.copy()
    
    # Veto words
    if 'veto_words' not in st.session_state:
        st.session_state.veto_words = DEFAULT_VETO_WORDS.copy()
    
    # Sentiment analysis
    if 'sentiment_analyzer' not in st.session_state:
        st.session_state.sentiment_analyzer = None
    
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    
    if 'bert_model_loaded' not in st.session_state:
        st.session_state.bert_model_loaded = False
    
    # Model training
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = ['SVM']
    
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = 'Quick Mode'
    
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {
            'test_size': 0.2,
            'cv_folds': 5,
            'smote_k_neighbors': 5,
            'random_state': 42
        }
    
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    
    if 'sentiment_map' not in st.session_state:
        st.session_state.sentiment_map = None

# Initialize session state
initialize_session_state()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Indonesian Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MAIN PAGE CONTENT
# ============================================================================

st.title("📊 Indonesian Sentiment Analysis Dashboard")
st.markdown("### Analyze sentiment in Indonesian text using BERT and ML models")

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Welcome! 👋
    
    This dashboard provides a comprehensive sentiment analysis toolkit for Indonesian text,
    specifically designed for analyzing public opinion on the **Makan Bergizi Gratis (MBG)** program.
    
    ### Features:
    
    1. **📁 Data Upload & Preprocessing**
       - Upload CSV files or use sample data
       - Advanced text cleaning with customizable options
       - Normalization dictionary management
       - Text statistics and visualization
    
    2. **🤖 Sentiment Analysis**
       - BERT-based sentiment classification
       - Custom rule-based refinement
       - Veto word system for enhanced accuracy
       - Confidence level categorization
    
    3. **🔧 Model Training**
       - Train SVM and KNN classifiers
       - Hyperparameter optimization with GridSearchCV
       - SMOTE oversampling for class balance
       - Comprehensive performance metrics
    
    4. **📈 Visualizations**
       - Interactive charts and graphs
       - Word clouds by sentiment
       - Model performance comparison
       - ROC curves and confusion matrices
    
    5. **🎯 Live Predictor**
       - Real-time sentiment prediction
       - Batch processing capability
       - Detailed prediction breakdown
    """)

with col2:
    st.markdown("### 🚀 Quick Start")
    
    st.info("""
    **Step 1:** Upload Data
    
    Go to the **📁 Data Upload** page to upload your CSV file or use sample data.
    """)
    
    st.info("""
    **Step 2:** Configure & Clean
    
    Set preprocessing options and run text cleaning.
    """)
    
    st.info("""
    **Step 3:** Analyze Sentiment
    
    Use BERT model to analyze sentiment with custom rules.
    """)
    
    st.info("""
    **Step 4:** Train Models
    
    Optionally train ML models for comparison.
    """)
    
    st.info("""
    **Step 5:** Visualize
    
    Explore results through interactive visualizations.
    """)

st.markdown("---")

# System Status
st.markdown("### 📊 Current Session Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.df_original is not None:
        st.success(f"✓ Data Loaded\n\n{len(st.session_state.df_original)} rows")
    else:
        st.warning("○ No Data\n\nUpload data first")

with col2:
    if st.session_state.df_cleaned is not None:
        st.success(f"✓ Text Cleaned\n\n{len(st.session_state.df_cleaned)} rows")
    else:
        st.warning("○ Not Cleaned\n\nRun preprocessing")

with col3:
    if st.session_state.df_sentiment is not None:
        st.success(f"✓ Sentiment Analyzed\n\n{len(st.session_state.df_sentiment)} rows")
    else:
        st.warning("○ Not Analyzed\n\nRun sentiment analysis")

with col4:
    if st.session_state.trained_models:
        st.success(f"✓ Models Trained\n\n{len(st.session_state.trained_models)} models")
    else:
        st.warning("○ No Models\n\nOptional training")

st.markdown("---")

# Navigation Guide
st.markdown("### 📚 Navigation Guide")

tabs = st.tabs(["Data Flow", "Key Concepts", "Tips & Tricks"])

with tabs[0]:
    st.markdown("""
    #### Complete Data Flow
    
    ```
    📁 Upload CSV/Sample Data
        ↓
    🔧 Configure Preprocessing
        ↓
    ✨ Clean & Normalize Text
        ↓
    🤖 Load BERT Model
        ↓
    🎯 Run Sentiment Analysis (with custom rules)
        ↓
    📊 View Results & Statistics
        ↓
    🔧 (Optional) Train ML Models
        ↓
    📈 Explore Visualizations
        ↓
    💾 Export Results
    ```
    
    Each step builds on the previous one, so follow the sequence for best results!
    """)

with tabs[1]:
    st.markdown("""
    #### Key Concepts
    
    **🎯 Veto Words**
    - Strong negative indicators that override BERT predictions
    - Example: "korupsi", "gagal", "bohong"
    - Immediately classifies text as negative
    
    **📊 Confidence Levels**
    - **High:** BERT score ≥ 0.80
    - **Medium:** BERT score 0.60-0.79
    - **Low:** BERT score < 0.60
    
    **🔄 Sentiment Mapping Logic**
    1. Check veto words (highest priority)
    2. Compare positive vs negative keyword counts
    3. Fall back to BERT prediction
    
    **⚖️ SMOTE Oversampling**
    - Balances class distribution in training data
    - Creates synthetic minority class samples
    - Improves model performance on imbalanced datasets
    """)

with tabs[2]:
    st.markdown("""
    #### Tips & Tricks
    
    **📁 Data Upload**
    - Use UTF-8 encoded CSV files
    - Ensure text column has no missing values
    - Start with sample data to explore features
    
    **🔧 Preprocessing**
    - Enable stemming for better word matching
    - Customize dictionary for domain-specific terms
    - Adjust minimum text length based on your data
    
    **🤖 Sentiment Analysis**
    - Review veto words carefully - they have highest priority
    - Monitor confidence distribution for model reliability
    - Export results for further analysis
    
    **🔧 Model Training**
    - Use Quick Mode for initial exploration
    - Full Analysis for production models
    - Compare multiple models before choosing
    
    **📈 Visualizations**
    - Use filters to focus on specific segments
    - Download charts as PNG for reports
    - Explore word clouds for keyword insights
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>Indonesian Sentiment Analysis Dashboard v1.0</strong></p>
    <p>Built with Streamlit • BERT • Scikit-learn • Plotly</p>
    <p>Specialized for MBG (Makan Bergizi Gratis) Program Analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.markdown("### 📌 Quick Links")
    st.markdown("""
    - [📁 Data Upload](pages/1_📁_Data_Upload.py)
    - [🤖 Sentiment Analysis](pages/2_🤖_Sentiment_Analysis.py)
    - [🔧 Model Training](pages/3_🔧_Model_Training.py)
    - [📈 Visualizations](pages/4_📈_Visualizations.py)
    - [🎯 Live Predictor](pages/6_🎯_Live_Predictor.py)
    """)
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This dashboard analyzes Indonesian text sentiment using:
    - **BERT Model**: ayameRushia/bert-base-indonesian
    - **Custom Rules**: Veto words & keyword matching
    - **ML Models**: SVM & KNN classifiers
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔄 Session Actions")
    
    if st.button("🗑️ Clear All Data", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['preprocessing_options', 'normalization_dict', 
                          'veto_words', 'model_params', 'analysis_mode']:
                del st.session_state[key]
        initialize_session_state()
        st.success("✓ All data cleared!")
        st.rerun()
    
    if st.button("🔄 Reset Settings", use_container_width=True):
        st.session_state.preprocessing_options = {
            'use_stemming': True,
            'use_stopwords': True,
            'min_text_length': 15
        }
        st.session_state.normalization_dict = DEFAULT_NORMALIZATION_DICT.copy()
        st.session_state.veto_words = DEFAULT_VETO_WORDS.copy()
        st.session_state.model_params = {
            'test_size': 0.2,
            'cv_folds': 5,
            'smote_k_neighbors': 5,
            'random_state': 42
        }
        st.session_state.analysis_mode = 'Quick Mode'
        st.success("✓ Settings reset to defaults!")
        st.rerun()