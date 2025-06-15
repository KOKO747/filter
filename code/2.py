import os
import re
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

output_root = "output"
metrics_to_plot = ['r2', 'adj_r2', 'mae', 'mape', 'rmse']
sets = ['train', 'val', 'test']


def parse_metrics(file_path):
    metrics = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"(\w+)_([\w]+)\s*:\s*([\d.]+)", line)
            if match:
                set_name, metric, value = match.groups()
                key = f"{set_name}_{metric}"
                metrics[key] = float(value)
    return metrics


round_dirs = sorted([d for d in os.listdir(output_root) if d.startswith("round")], key=lambda x: int(x.replace("round", "")))

all_metrics = {f"{s}_{m}": [] for s in sets for m in metrics_to_plot}
round_indices = []

for d in round_dirs:
    round_num = int(d.replace("round", ""))
    file_path = os.path.join(output_root, d, f"model_evaluation_round{round_num}.txt")
    if os.path.exists(file_path):
        round_indices.append(round_num)
        metrics = parse_metrics(file_path)
        for key in all_metrics:
            all_metrics[key].append(metrics.get(key, None))


os.makedirs("analysis_plots", exist_ok=True)

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    for set_name in sets:
        key = f"{set_name}_{metric}"
        plt.plot(round_indices, all_metrics[key], marker='o', label=set_name.capitalize())
    plt.title(f"{metric.upper()} over Rounds")
    plt.xlabel("Round")
    plt.ylabel(metric.upper())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"analysis_plots/{metric}_trend.png", dpi=600)
    plt.close()

print("analysis_plots/")

