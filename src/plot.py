# show table of the results
import tabulate
import json
import os

results = []
for file in os.listdir("eval_results"):
    model_name = file.split("_")[-1]
    json_data = json.load(open("eval_results/" + file))
    score = sum([item['score'] for item in json_data]) / len(json_data)
    results.append([model_name, score * 100])

sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
print(tabulate.tabulate(sorted_results, headers=["Model", "Score"], tablefmt="fancy_grid", floatfmt=".2f"))
    