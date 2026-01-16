from openai import OpenAI
import os
import json
import time
import streamlit as st
from schema import get_schema
from dotenv import load_dotenv
from utils import read_json
from datasets import load_dataset
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
    for i in range(max_retries):
        predictions = {}
        error = None
        prompt, sys_prompt = schema.get_prompts(text_input, readme = "", version = version)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]

        
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
            response =  message.choices[0].message.content
            predictions = read_json(response)
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
    if predictions == {}:
        predictions = schema.generate_metadata(method = 'default').json()
    return message, predictions, error

def compare(metadata, pred_metadata):
    schema = get_schema("ar")
    gold_metadata = schema(metadata = metadata)
    score = gold_metadata.compare_with(pred_metadata, return_metrics_only = True)
    return {"f1": score["f1"]}

        
def add_annotations(pred_metadata):
    pred_metadata["annotations_from_paper"] = {}
    for key in pred_metadata.keys():
        if key in ['annotations_from_paper']:
            continue
        pred_metadata["annotations_from_paper"][key] = 1
    return pred_metadata        

def main():
    st.set_page_config(
        page_title="Masader Chat",
        page_icon="🤖",
        layout="wide"
    )
    dataset = load_dataset("src/masader")
    
    st.title("🤖 Masader Chat")
    st.markdown("Extract metadata from dataset descriptions using AI models")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=["moonshotai/kimi-k2", "qwen3", "nuextract"],
        index=0
    )
    
    # Schema selection
    schema_name = st.sidebar.selectbox(
        "Select Schema",
        options=["ar", "en"],
        index=0
    )
    
    # Backend selection
    backend = st.sidebar.selectbox(
        "Select Backend",
        options=["openrouter"],
        index=0
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        max_retries = st.slider("Max Retries", min_value=1, max_value=5, value=3)
        timeout = st.slider("Timeout (seconds)", min_value=1, max_value=10, value=3)
        version = st.selectbox("Version", options=["1.0", "2.0"], index=1)
    
    # Main interface
    st.header("Input")
    
    # Text input area
    text_input = st.text_area(
        "Enter dataset description or name:",
        placeholder="e.g., PEARL dataset",
        height=100
    )
    
    # Process button
    if st.button("🚀 Process", type="primary"):
        if not text_input.strip():
            st.error("Please enter some text to process.")
            return
        
        # Show processing status
        with st.spinner("Processing..."):
            try:
                message, predictions, error = get_metadata(
                    text_input=text_input,
                    model_name=model_name,
                    schema_name=schema_name,
                    max_retries=max_retries,
                    backend=backend,
                    timeout=timeout,
                    version=version
                )
                
                # Display results
                st.header("Results")
                
                # Show error if any
                if error:
                    st.error(f"Error: {error}")
                
                # Show predictions
                if predictions:
                    st.subheader("📊 Extracted Metadata")
                    st.json(predictions)
                    pred_metadata = add_annotations(predictions)                    
                    dataset = dataset.map(compare, fn_kwargs={"pred_metadata": pred_metadata})
                    sorted_dataset = dataset.sort("f1", reverse=True)
                    for dataset in sorted_dataset['train']:
                        f1_score = dataset['f1']
                        if f1_score > 0.7:
                            st.write(dataset['Name'], dataset["Link"], f1_score)
                        else:
                            break
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

    