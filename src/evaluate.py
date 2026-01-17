from db import DatasetsDatabase
import json
from tqdm import tqdm
from utils import get_metadata
import concurrent.futures
from typing import Dict, Any, Tuple

def process_eval_item(eval_item: Dict[str, Any], model_name: str, db: DatasetsDatabase) -> float:
    """Process a single evaluation item and return its score."""
    try:
        message, query, error = get_metadata(eval_item["query"], model_name=model_name)
        if error:
            print(f"Error processing query '{eval_item['query']}': {error}")
            return 0.0
            
        response = db.query(query)
        ids = [dataset["id"] for dataset in response]
        pred_ids = set(ids)
        eval_ids = set(eval_item["ids"])
        score = len(pred_ids & eval_ids) / max(len(pred_ids), len(eval_ids))
        return score
    except Exception as e:
        print(f"Exception processing query '{eval_item['query']}': {str(e)}")
        return 0.0

def evaluate(db: DatasetsDatabase, model_name: str = "google/gemini-3-pro-preview", max_workers: int = 5) -> float:
    """Evaluate the model's performance on the evaluation set with parallel processing."""
    with open("src/evals.json", "r") as f:
        evals = json.load(f)
    
    total_score = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each evaluation item
        future_to_eval = {
            executor.submit(process_eval_item, eval_item, model_name, db): eval_item 
            for eval_item in evals
        }
        
        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_eval),
            total=len(evals),
            desc="Processing evaluations"
        ):
            eval_item = future_to_eval[future]
            try:
                score = future.result()
                total_score += score
            except Exception as e:
                print(f"Exception in future for query '{eval_item['query']}': {str(e)}")
    
    avg_accuracy = total_score / len(evals) if evals else 0.0
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy

if __name__ == "__main__":
    db = DatasetsDatabase()
    evaluate(db, max_workers=1)