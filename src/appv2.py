
from schema import get_schema
import os
import json
import time
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
    You need to return the id, Name and Link of the dataset. Return the SQL query ONLY, do not return any additional text.
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

def evaluate(db, model_name = "google/gemini-3-pro-preview"):
    with open("src/evals.json", "r") as f:
        evals = json.load(f)
    accuracy = 0
    for eval in tqdm(evals):
        message, query, error = get_metadata(eval["query"], model_name=model_name)
        response = db.query(query)
        ids = [dataset["id"] for dataset in response]
        pred_ids = set(ids)
        eval_ids = set(eval["ids"])
        score = len(pred_ids & eval_ids)/ max(len(pred_ids), len(eval_ids))
        accuracy += score
    print("Accuracy ", accuracy/ len(evals))
        
        

if __name__ == "__main__":
    # message, query, error = get_metadata("datasets that contain the Egypt Dialect", model_name="anthropic/claude-opus-4.5")
    # print(query)
    db = DatasetsDatabase()
    # response = db.query(query)
    # for dataset in response:
    #     print(dataset["Name"], dataset["Link"])
    evaluate(db, model_name = "anthropic/claude-opus-4.5")

### Problems:
### 1. when searching for the dialects it doesn't consider the Subsets column
### 2. seems anthropic/claude-opus-4.5 is the best for now 
### 3. I need to create a dataset for evaluation