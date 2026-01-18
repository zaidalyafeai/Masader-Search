import json 
import os
from openai import OpenAI
from schema import get_schema
from dotenv import load_dotenv
import time
from db import DatasetsDatabase
load_dotenv()
def get_metadata(
    text_input,
    model_name="moonshotai/kimi-k2",
    schema_name="ar",
    max_retries = 3,
    backend = "openrouter",
    timeout = 3,
):
    db = DatasetsDatabase()
    schema = get_schema(schema_name)
    # keys = list(json.loads(schema.schema()).keys()) 
    system_prompt = f"""
    You are a helpful assistant that generates SQL queries (using python sqlite3) based on a given text input. The database contains 
    datasets with the following schema: {schema.schema()}, the table name is DATASETS. Each key in the schema represents a column in the table. 
    You need to return the id, Name of the dataset. Return the SQL query ONLY, do not return any additional text.
    """
    prompt = text_input
    predictions = {}
    for _ in range(max_retries):
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

        
        message = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        try:
            predictions =  message.choices[0].message.content
            predictions = process_sql(predictions)
            db.query(predictions)
        except Exception as e:
            if message is None:
                error = "Timeout"
            elif message.choices is None:
                error = message.error["message"]
            else:
                error = str(e)
            print(error)
        if error is None:
            break
        time.sleep(timeout)
    return predictions, error

def removeStartAndEndQuotes(json_str):
    if json_str.startswith('"') and json_str.endswith('"'):
        print("fixing")
        return json_str[1:-1]
    else:
        return json_str

def process_sql(sql_str: str) -> str:
    sql_str = sql_str.replace("```sql", "").replace("```", "")
    return sql_str.strip()

def singleQuoteToDoubleQuote(singleQuoted):
    """
    convert a single quoted string to a double quoted one
    Args:
        singleQuoted(string): a single quoted string e.g. {'cities': [{'name': "Upper Hell's Gate"}]}
    Returns:
        string: the double quoted version of the string e.g.
    see
        - https://stackoverflow.com/questions/55600788/python-replace-single-quotes-with-double-quotes-but-leave-ones-within-double-q
    """
    cList = list(singleQuoted)
    inDouble = False
    inSingle = False
    for i, c in enumerate(cList):
        # print ("%d:%s %r %r" %(i,c,inSingle,inDouble))
        if c == "'":
            if not inDouble:
                inSingle = not inSingle
                cList[i] = '"'
        elif c == '"':
            inDouble = not inDouble
    doubleQuoted = "".join(cList)
    return doubleQuoted

def fix_json(json_str: str) -> str:
    """
    Attempts to fix common issues in a malformed JSON string.

    Args:
        broken_json (str): The malformed JSON string.

    Returns:
        str: The corrected JSON string if fixable, or an error message.
    """
    try:
        # remove \escaping cahracters
        json_str = json_str.replace("\\", "")
        # remove start and end quotes
        json_str = removeStartAndEndQuotes(json_str)
        # replace single quotes to double quotes
        json_str = singleQuoteToDoubleQuote(json_str)

        loaded_json = json.loads(json_str)

        return loaded_json
    except json.JSONDecodeError as e:
        raise e


def read_json(text_json):
    text_json = text_json.replace("```json", "").replace("```", "")
    fixed_json = fix_json(text_json)
    if 'answer' in fixed_json:
        fixed_json = fixed_json['answer']
    return fixed_json

