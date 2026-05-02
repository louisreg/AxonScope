import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import math

def plot_benchmark(file="benchmark.csv"):
    """
    Plot execution time vs problem size from a benchmark CSV file.
    Highlights 'nrv_neuron' label and formats legend on multiple columns.
    """
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, file)
    df = pd.read_csv(full_path)

    # Seaborn style
    sns.set_theme(style="whitegrid", context="talk")

    # Large figure for many labels
    plt.figure(figsize=(15, 10))

    # Palette : des couleurs sobres
    palette = sns.color_palette("tab20", n_colors=df["label"].nunique())

    # Plot each label separately
    for label in df["label"].unique():
        data = df[df["label"] == label]
        if label == "nrv_neuron":
            plt.plot(
                data["N"],
                data["time"],
                marker="o",
                color="black",
                linewidth=3,
                markersize=8,
                label=label,
                zorder=10,
            )
        else:
            plt.plot(
                data["N"],
                data["time"],
                marker="o",
                linewidth=1.5,
                label=label,
                alpha=0.7,
                zorder=5,
            )

    # Axes scaling
    plt.xscale("log")
    plt.yscale("log")

    # Labels & title
    plt.xlabel("Spatial Points", fontsize=14)
    plt.ylabel("Execution Time (s)", fontsize=14)
    plt.title("Benchmark Results", fontsize=16, weight="bold")

    # Grid & layout
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if "nrv_neuron" in labels:
        idx = labels.index("nrv_neuron")
        # Reorder legend to put "nrv_neuron" first
        handles = [handles[idx]] + [h for i, h in enumerate(handles) if i != idx]
        labels = ["nrv_neuron"] + [l for l in labels if l != "nrv_neuron"]

    # Automatically decide number of legend columns (max 3)
    n_labels = len(labels)
    ncols = min(3, math.ceil(n_labels / 7))

    legend = plt.legend(
        handles,
        labels,
        title="Implementation",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2
                        ),
        ncol=ncols,
        fontsize=10,
        frameon=True,
    )

    for text in legend.get_texts():
        if text.get_text() == "nrv_neuron":
            text.set_fontweight("bold")
            text.set_color("black")

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for legend below
    plt.show()


if __name__ == "__main__":
    plot_benchmark()
