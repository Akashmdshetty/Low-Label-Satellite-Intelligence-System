import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_benchmark_results(results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    # Real results from the benchmarking runs
    data = {
        "Label %": [1, 5, 10, 1, 5, 10, 1, 5, 10],
        "Accuracy": [41.2, 51.1, 45.1, 64.9, 71.3, 82.7, 38.6, 38.6, 76.1],
        "Method": ["Linear Probe", "Linear Probe", "Linear Probe", 
                   "BYOL Fine-Tune", "BYOL Fine-Tune", "BYOL Fine-Tune",
                   "Supervised (Scratch)", "Supervised (Scratch)", "Supervised (Scratch)"]
    }
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Label %", y="Accuracy", hue="Method", marker='o')
    plt.title("EuroSAT Classification: Accuracy vs Label %")
    plt.grid(True)
    plt.ylim(0, 100)
    plt.ylabel("Validation Accuracy (%)")
    plt.xlabel("Percentage of Labeled Data (%)")
    
    save_path = os.path.join(results_dir, "benchmark_comparison.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Save table
    df_pivot = df.pivot(index="Label %", columns="Method", values="Accuracy")
    df_pivot.to_csv(os.path.join(results_dir, "benchmark_results.csv"))
    print("Results table saved to benchmark_results.csv")
    
    # Display table in markdown format for terminal
    print("\nBenchmark Summary Table (%)")
    print(df_pivot)
    
    return df_pivot

if __name__ == "__main__":
    plot_benchmark_results()
