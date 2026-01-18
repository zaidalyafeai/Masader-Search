# show table of the results
import tabulate
import json
import os

results = []
for file in os.listdir("eval_results"):
    model_name = file.split("_")[-1]
    json_data = json.load(open("eval_results/" + file))
    score = json_data['average_score'] * 100
    results.append([model_name, score])

sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
print(tabulate.tabulate(sorted_results, headers=["Model", "Score"], tablefmt="fancy_grid", floatfmt=".2f"))
    