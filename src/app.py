import streamlit as st

# Set page config
st.set_page_config(
    page_title="Masader Chat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Add some custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 200px;
        max-width: 200px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0 0;
        gap: 8px;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)
