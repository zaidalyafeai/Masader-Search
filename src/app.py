
from schema import get_schema
import os
import json
import time
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from db import DatasetsDatabase
from utils import process_sql
from tqdm import tqdm

load_dotenv()
def get_metadata(
    text_input,
    model_name="moonshotai/kimi-k2",
    schema_name="ar",
    max_retries = 3,
    backend = "openrouter",
    timeout = 3,
    version = "2.0",
):
    schema = get_schema(schema_name)
    # keys = list(json.loads(schema.schema()).keys()) 
    system_prompt = f"""
    You are a helpful assistant that generates SQL queries (using sqlite3) based on a given text input. The database contains 
    datasets with the following schema: {schema.schema()}, the table name is DATASETS. Each key in the schema represents a column in the table. 
    You need to return the id, Name of the dataset. Return the SQL query ONLY, do not return any additional text.
    """
    prompt = text_input
    for i in range(max_retries):
        predictions = {}
        error = None
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        
        if backend == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ValueError(f"Invalid backend: {backend}")

        if 'nuextract' in model_name.lower():
            template = schema.schema_to_template()
            message = client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={
                "chat_template_kwargs": {
                    "template": json.dumps(json.loads(template), indent=4)
                },
            })
        else:
            if "qwen3" in model_name.lower():
                message = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                    )
            else:
                message = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                    )
        try:
            predictions =  message.choices[0].message.content
            predictions = process_sql(predictions)
        except json.JSONDecodeError as e:
            error = str(e)
        except Exception as e:
            if message is None:
                error = "Timeout"
            elif message.choices is None:
                error = message.error["message"]
            else:
                error = str(e)
            print(error)
        if predictions != {}:
            break
        else:
            pass
    time.sleep(timeout)
    return message, predictions, error


@st.cache_resource
def _get_db() -> DatasetsDatabase:
    return DatasetsDatabase()


def _render_app() -> None:
    st.set_page_config(page_title="Masader Search", layout="wide")

    def load_css(file_name: str):
        """A function to load a css file."""
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.markdown("<h1>Masader Search</h1>", unsafe_allow_html=True)

    st.markdown("<h4>Examples:</h4>", unsafe_allow_html=True)
    examples = {
        "Audio datasets (more than 1000 hours)": "SELECT id, Name FROM DATASETS WHERE Form='audio' AND Volume > 1000 AND Unit='hours'",
        "Permissive licensed datasets": "SELECT id, Name FROM DATASETS WHERE License <> 'custom' AND License NOT LIKE '%LDC%' AND License NOT LIKE '%ELRA%' AND License <> 'unknown'",
        "HuggingFace hosted datasets": "SELECT id, Name FROM DATASETS WHERE Host='HuggingFace' or Link LIKE '%huggingface%'",
    }

    if 'query' not in st.session_state:
        st.session_state.query = ""

    cols = st.columns(len(examples))
    for i, (key, value) in enumerate(examples.items()):
        if cols[i].button(key, type="secondary"):
            st.session_state.query = key

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    query_text = st.text_input(
        label="",
        placeholder="Type your query here (e.g., datasets that contain the Egypt Dialect)",
        value=st.session_state.query,
    )

    left, mid, right = st.columns([1, 1, 1])
    with mid:
        run = st.button("Get metadata", type="primary", use_container_width=True)

    model_name = "anthropic/claude-opus-4.5"
    schema_name = "ar"

    if run or st.session_state.query:
        if not query_text.strip():
            st.warning("Please enter a query.")
            st.session_state.query = ""
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
        
        st.session_state.query = ""

        with st.spinner("Querying the database..."):
            st.subheader("Generated SQL")
            st.code(sql_query or "", language="sql")


            if not sql_query or not str(sql_query).strip():
                st.warning("No SQL was generated.")
                return

            db = _get_db()
            rows = db.query(sql_query)

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
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.LinkColumn("id", display_text=r".*id=([0-9]+)$"),
            },
        )


_render_app()

### Problems:
### 1. when searching for the dialects it doesn't consider the Subsets column
### 2. seems anthropic/claude-opus-4.5 is the best for now 
### 3. I need to create a dataset for evaluation