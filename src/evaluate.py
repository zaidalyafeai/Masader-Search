from db import DatasetsDatabase
import json
from tqdm import tqdm
from appv2 import get_metadata

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
    db = DatasetsDatabase()
    evaluate(db)