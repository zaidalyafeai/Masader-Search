from db import DatasetsDatabase
import json
from tqdm import tqdm
from utils import get_metadata
import concurrent.futures
from typing import Dict, Any
import argparse
import sqlite3
import os

def get_ids_from_sql(sql: str, db: DatasetsDatabase) -> set:
    response = db.query(sql)
    ids = [dataset["id"] for dataset in response]
    return set(ids)
    
def process_eval_item(eval_item: Dict[str, Any], model_name: str, db_path: str) -> Dict[str, Any]:
    """Process a single evaluation item with its own database connection and cursor."""
    query, error = get_metadata(eval_item["query"], model_name=model_name)
    try:
        # Create a new database connection for this thread
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Helper function to execute SQL with the local cursor
            def get_ids(sql: str) -> set:
                cursor.execute(sql)
                return {row["id"] for row in cursor.fetchall()}
            
            pred_ids = get_ids(query)
            eval_ids = get_ids(eval_item["sql"])
            score = len(pred_ids & eval_ids) / max(len(pred_ids), len(eval_ids), 1)  # Avoid division by zero
            return {
                "id": eval_item["id"],
                "score": score,
                "query": eval_item["query"],
                "pred_sql": query,
                "gold_sql": eval_item["sql"]
            }
    except Exception as e:
        print(f"Exception processing query '{eval_item['query']}': {str(e)}")
        return {
            "id": eval_item["id"],
            "score": 0.0,
            "query": eval_item["query"],
            "pred_sql": query,
            "gold_sql": eval_item["sql"]
        }

def evaluate(db_path: str, model_name: str = "google/gemini-3-pro-preview", max_workers: int = 5) -> float:
    """Evaluate the model's performance on the evaluation set with parallel processing."""
    with open("evals.json", "r") as f:
        evals = json.load(f)
    
    total_score = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each evaluation item
        future_to_eval = {
            executor.submit(process_eval_item, eval_item, model_name, db_path): eval_item 
            for eval_item in evals
        }
        
        # Process results as they complete
        save_path = "eval_results/eval_results_{}.json".format(model_name.split("/")[-1])
        results = []
        for future in tqdm(
            concurrent.futures.as_completed(future_to_eval),
            total=len(evals),
            desc="Processing evaluations"
        ):
            result = future.result()
            total_score += result["score"]
            results.append(result)

    
    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "model": model_name,
            "average_score": total_score / len(evals) if evals else 0,
            "results": results
        }, f, indent=2)
    
    return total_score / len(evals) if evals else 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance on the evaluation set.")
    parser.add_argument("--model_name", type=str, default="google/gemini-3-pro-preview", help="Model name to evaluate")
    parser.add_argument("--max_workers", type=int, default=5, help="Number of workers to use for parallel processing")
    parser.add_argument("--db_path", type=str, default="datasets.db", help="Path to the SQLite database")
    args = parser.parse_args()
    
    score = evaluate(
        db_path=args.db_path,
        model_name=args.model_name,
        max_workers=args.max_workers
    )
    print(f"Average score: {score:.4f}")