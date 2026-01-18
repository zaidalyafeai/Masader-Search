from db import DatasetsDatabase
import json
from tqdm import tqdm
from utils import get_metadata
import concurrent.futures
from typing import Dict, Any
import argparse

def get_ids_from_sql(sql: str, db: DatasetsDatabase) -> set:
    response = db.query(sql)
    ids = [dataset["id"] for dataset in response]
    return set(ids)
    
def process_eval_item(eval_item: Dict[str, Any], model_name: str, db: DatasetsDatabase) -> float:
    """Process a single evaluation item and return its score."""
    try:
        message, query, error = get_metadata(eval_item["query"], model_name=model_name)
        if error:
            print(f"Error processing query '{eval_item['query']}': {error}")
            return 0.0
            
        pred_ids = get_ids_from_sql(query, db)
        eval_ids = get_ids_from_sql(eval_item["sql"], db)
        score = len(pred_ids & eval_ids) / max(len(pred_ids), len(eval_ids))
        return {"id": eval_item["id"], "score": score, "query": eval_item["query"], "pred_sql": query, "gold_sql": eval_item["sql"]}
    except Exception as e:
        print(f"Exception processing query '{eval_item['query']}': {str(e)}")
        return {"id": eval_item["id"], "score": 0.0, "query": eval_item["query"], "pred_sql": None, "gold_sql": eval_item["sql"]}

def evaluate(db: DatasetsDatabase, model_name: str = "google/gemini-3-pro-preview", max_workers: int = 5) -> float:
    """Evaluate the model's performance on the evaluation set with parallel processing."""
    with open("evals.json", "r") as f:
        evals = json.load(f)
    
    total_score = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each evaluation item
        future_to_eval = {
            executor.submit(process_eval_item, eval_item, model_name, db): eval_item 
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
            eval_item = future_to_eval[future]
            try:
                result = future.result()
                total_score += result["score"]
                results.append(result)
            except Exception as e:
                print(f"Exception in future for query '{eval_item['query']}': {str(e)}")
    
    avg_accuracy = total_score / len(evals) if evals else 0.0
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    sorted_results = sorted(results, key=lambda x: x["id"])
    with open(save_path, "w") as f:
        json.dump(sorted_results, f, indent=4)
    return avg_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance on the evaluation set.")
    parser.add_argument("--model_name", type=str, default="google/gemini-3-pro-preview", help="Model name to evaluate")
    parser.add_argument("--max_workers", type=int, default=5, help="Number of workers to use for parallel processing")
    args = parser.parse_args()
    db = DatasetsDatabase()
    evaluate(db, max_workers=args.max_workers, model_name=args.model_name)