import streamlit as st
import pandas as pd
from db import DatasetsDatabase

@st.cache_resource
def get_db():
    return DatasetsDatabase()

def _render_app():
    # Title and description
    st.title("📊 Query Analytics")
    st.markdown("""
    View recent search queries made in the Masader Search application.
    """)

    db = get_db()
    
    # Get recent queries
    recent_queries = db.get_recent_queries(limit=50)
    
    if not recent_queries:
        st.info("No recent queries found.")
    else:
        df_recent = pd.DataFrame(recent_queries)
        
        # Format timestamp for display
        if not df_recent.empty and 'timestamp' in df_recent.columns:
            df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'])
            df_recent['time_ago'] = df_recent['timestamp'].apply(
                lambda x: f"{pd.Timestamp.now() - x}".split('.')[0] + ' ago'
            )
        
        # Display recent queries in a table
        st.dataframe(
            df_recent[['natural_language_query', 'sql_query', 'response_count', 'time_ago']].rename(
                columns={
                    'natural_language_query': 'Natural Language Query',
                    'sql_query': 'SQL Query',
                    'response_count': 'Results',
                    'time_ago': 'When'
                }
            ),
            hide_index=True
        )

    # Add some custom CSS
    st.markdown("""
    <style>
        .stDataFrame {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# This ensures the _render_app function is called when the module is imported
_render_app()
