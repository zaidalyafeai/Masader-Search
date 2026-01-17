
import os
import streamlit as st
import pandas as pd
import time
from db import DatasetsDatabase
from utils import get_metadata
from rate_limiter import RateLimiter, get_client_ip

@st.cache_resource
def _get_db() -> DatasetsDatabase:
    return DatasetsDatabase()


def _render_app() -> None:
    # Initialize rate limiter: 10 requests per minute per IP
    if 'rate_limiter' not in st.session_state:
        st.session_state.rate_limiter = RateLimiter(requests=5, time_window=60)  # 10 requests per minute
    
    rate_limiter = st.session_state.rate_limiter
    client_ip = get_client_ip()
    
    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        retry_after = rate_limiter.get_retry_after(client_ip)
        st.error(f"Rate limit exceeded. Please try again in {retry_after} seconds.")
        return

    st.markdown("""
    <style>
        input {color: #0054a3 !important;}
    </style>
    """, unsafe_allow_html=True)
    "# 📮 :rainbow[Masader Search]"

    st.info(
        """
        This is an enhanced Masader Search that allows users to find datasets in Masader using natural language.
        It uses an LLM to generate the SQL query from a given natural text prompt. This tool is most useful
        for queries that are deterministic and simple. If you face any issues, please report them on [GitHub](https://github.com/zaidalyafeai/Masader-Search).
        """,
        icon="ℹ️",
    )

    def load_css(file_name: str):
        """A function to load a css file."""
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    st.markdown("<h4>Examples:</h4>", unsafe_allow_html=True)
    examples = {
        "Audio datasets (more than 1000 hours)": "SELECT id, Name FROM DATASETS WHERE Form='audio' AND Volume > 1000 AND Unit='hours'",
        "Permissive licensed datasets": "SELECT id, Name FROM DATASETS WHERE License <> 'custom' AND License NOT LIKE '%LDC%' AND License NOT LIKE '%ELRA%' AND License <> 'unknown'",
        "Datasets for language modelling > 100 billion tokens": "SELECT id, Name FROM DATASETS WHERE Tasks LIKE '%language modeling%' AND Volume > 100000000000",
    }


    cols = st.columns(len(examples))

    for i, (key, value) in enumerate(examples.items()):
        if cols[i].button(key, type="secondary"):
            st.session_state.query_input = key
    
    query_text = st.text_input(
        label="Input",
        placeholder="Type your query here (e.g., datasets that contain the Egyptian Dialect)",
        key="query_input",
    )

    # st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    

    left, mid, right = st.columns([1, 1, 1])
    with mid:
        run = st.button("Search Masader", type="primary")

    model_name = "moonshotai/kimi-k2"
    schema_name = "ar"

    if run:
        if not query_text.strip():
            st.warning("Please enter a query.")
            return
        elif len(query_text.split(' ')) > 100:
            st.warning("Query is too long. Please enter a shorter query.")
            return
        elif len(query_text) > 1000:
            st.warning("Query is too long. Please enter a shorter query.")
            return

        if "SELECT" in query_text.upper():
            sql_query = query_text
            _message, error = None, None
        else:
            if not os.environ.get("OPENROUTER_API_KEY"):
                st.error("Missing OPENROUTER_API_KEY in your environment.")
                return
            with st.spinner("Generating SQL..."):
                if query_text in examples:
                    sql_query = examples[query_text]
                else:
                    _message, sql_query, error = get_metadata(
                        query_text,
                        model_name=model_name,
                        schema_name=schema_name,
                    )
        

        with st.spinner("Querying the database..."):
            st.subheader("SQL Query")
            st.code(sql_query or "", language="sql")


            if not sql_query or not str(sql_query).strip():
                st.warning("No SQL was generated.")
                return

            db = _get_db()
            rows = db.query(sql_query)
            # Log both the natural language query and SQL query with result count
            db.log_query(
                natural_language_query=query_text,
                sql_query=sql_query,
                response_count=len(rows) if rows else 0
            )

        st.subheader("Datasets")
        st.write(f"Retrived {len(rows)} datasets")

        if not rows:
            st.info("No results.")
            return

        df = pd.DataFrame(rows)
        if "id" in df.columns:
            df["id"] = df["id"].apply(lambda x: f"https://arbml.github.io/masader/card?id={x}")

        st.dataframe(
            df,
            hide_index=True,
            column_config={
                "id": st.column_config.LinkColumn("id", display_text=r".*id=([0-9]+)$"),
            },
        )


_render_app()
