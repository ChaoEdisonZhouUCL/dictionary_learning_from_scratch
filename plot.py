import pandas as pd
import matplotlib.pyplot as plt


def plot_log_exp():
    # Read the CSV file
    df = pd.read_csv("log/nuc_norm.csv")

    num_cols = len(df.columns)

    line_styles = [
        ("solid", "-"),
        ("dashed", "--"),
        ("dashdot", "-."),
        ("dotted", ":"),
        ("none", ""),
    ]

    markers = [
        ("circle", "o"),
        ("pixel", ","),
        ("triangle_down", "v"),
        ("triangle_up", "^"),
        ("triangle_left", "<"),
        ("triangle_right", ">"),
        ("square", "s"),
        ("pentagon", "p"),
        ("star", "*"),
        ("hexagon1", "h"),
        ("hexagon2", "H"),
        ("plus", "+"),
        ("x", "x"),
        ("diamond", "D"),
        ("thin_diamond", "d"),
    ]

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Plot each column
    for idx in range(1, num_cols, 3):

        reg_value = df.columns[idx].split("=")[1].split(",")[0]
        print(
            df.columns[idx],
            reg_value,
            line_styles[(idx // 3) // 5][1],
            markers[(idx // 3) // 15][1],
        )

        plt.plot(
            df.index[::5],
            df.iloc[::5, idx],
            label=r"$\lambda=$" + reg_value,
            linestyle=line_styles[(idx // 3) // 5][1],
            marker=markers[idx // 3][1],
            linewidth=4,
            markersize=10,
        )

    # Customize the plot
    # plt.title("Training Process for Different Configurations")
    plt.grid(True)
    plt.xlim(left=0)
    plt.yscale("log")
    plt.xlabel("iterations", fontsize=24)
    plt.ylabel("nuclear norm of sparse code (log scale)", fontsize=24)
    plt.legend(fontsize=16)

    # Save the plot as an EPS file
    plt.savefig("log/nuc_norm.eps", format="eps", dpi=100, bbox_inches="tight")

    print("Plot saved as training_process.eps")

    # Optionally, display the plot
    plt.show()


def plot_power_exp():
    # Read the CSV file
    df = pd.read_csv("power/recon_MSE.csv")

    num_cols = len(df.columns)

    line_styles = [
        ("solid", "-"),
        ("dashed", "--"),
        ("dashdot", "-."),
        ("dotted", ":"),
        ("none", ""),
    ]

    markers = [
        ("circle", "o"),
        ("triangle_down", "v"),
        ("triangle_up", "^"),
        ("triangle_left", "<"),
        ("triangle_right", ">"),
        ("square", "s"),
        ("pentagon", "p"),
        ("star", "*"),
        ("hexagon1", "h"),
        ("hexagon2", "H"),
        ("plus", "+"),
        ("x", "x"),
        ("diamond", "D"),
        ("thin_diamond", "d"),
    ]

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Plot each column
    for idx in range(1, num_cols, 3):

        reg_value = df.columns[idx].split("=")[2].split(",")[0]
        print(
            df.columns[idx],
            reg_value,
            line_styles[(idx // 3) // 5][1],
            markers[(idx // 3) // 14][1],
        )

        plt.plot(
            df.index[::5],
            df.iloc[::5, idx],
            label=r"$n=$" + reg_value,
            linestyle=line_styles[(idx // 3) // 5][1],
            marker=markers[idx // 3][1],
            linewidth=4,
            markersize=10,
        )

    # Customize the plot
    # plt.title("Training Process for Different Configurations")
    plt.grid(True)

    # Set the x-axis to start from 0
    plt.xlim(left=0)
    plt.xlabel("iterations", fontsize=24)
    plt.ylabel("reconstruction MSE (log scale)", fontsize=24)
    plt.legend(fontsize=16, loc="best")
    # plt.xticks(fontsize=19)  # X-axis tick size
    # plt.yticks(fontsize=19)  # Y-axis tick size
    plt.yscale("log")
    # Save the plot as an EPS file
    plt.savefig("power/recon_MSE.eps", format="eps", dpi=100, bbox_inches="tight")

    print("Plot saved as training_process.eps")

    # Optionally, display the plot
    plt.show()


def plot_lora_ratio():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # Assuming you have 4 .pt files, each storing a dictionary of metrics

    file_paths = [
        "LORA/ratiowd0.001.pt",
        "LORA/ratiowd0.01.pt",
        "LORA/ratiowd0.1.pt",
        "LORA/ratiowd1.pt",
    ]
    line_styles = [
        ("solid", "-"),
        ("dashed", "--"),
        ("dashdot", "-."),
        ("dotted", ":"),
        ("none", ""),
    ]

    markers = [
        ("circle", "o"),
        ("pixel", ","),
        ("triangle_down", "v"),
        ("triangle_up", "^"),
        ("triangle_left", "<"),
        ("triangle_right", ">"),
        ("square", "s"),
        ("pentagon", "p"),
        ("star", "*"),
        ("hexagon1", "h"),
        ("hexagon2", "H"),
        ("plus", "+"),
        ("x", "x"),
        ("diamond", "D"),
        ("thin_diamond", "d"),
    ]
    # Dictionary to store metrics from all files
    all_metrics = {}

    for i, file_path in enumerate(file_paths):
        # Load the file
        metrics = torch.load(file_path, map_location=torch.device("cpu"))
        metrics_list = []
        for metric in metrics:
            metrics_list.append(metric.detach().numpy())
        # Assuming the .pt files contain metrics such as loss and accuracy
        wd = file_path.split("wd")[1].split(".pt")[0]
        all_metrics[f"wd={wd}"] = metrics_list

    # Example: Plot the "loss" vs. "epoch" for each run
    plt.figure(figsize=(12, 8))

    for i in range(4):
        name = list(all_metrics.keys())[i]
        metrics = all_metrics[name]
        Xs = np.linspace(1, len(metrics), len(metrics))
        plt.plot(
            Xs[::5],
            metrics[::5],
            label=name,
            linestyle=line_styles[i][1],
            marker=markers[i][1],
            linewidth=3,
            markersize=5,
        )

    # Customize the plot
    # plt.title("Training Process for Different Configurations")
    plt.grid(True)
    plt.xlim(left=0)
    # plt.yscale("log")
    # plt.title("ratio vs Epochs")
    plt.xlabel("Iterations", fontsize=24)
    plt.ylabel("Nuclear norm / Frobenius norm", fontsize=24)
    plt.legend(fontsize=16)

    # Save the plot as an EPS file
    plt.savefig("LORA/ratio.eps", format="eps", dpi=100, bbox_inches="tight")

    # Optionally, display the plot
    plt.show()


def plot_lora_acc():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # Assuming you have 4 .pt files, each storing a dictionary of metrics

    file_paths = [
        "LORA/valaccwd0.001.pt",
        "LORA/valaccwd0.01.pt",
        "LORA/valaccwd0.1.pt",
        "LORA/valaccwd1.pt",
    ]
    line_styles = [
        ("solid", "-"),
        ("dashed", "--"),
        ("dashdot", "-."),
        ("dotted", ":"),
        ("none", ""),
    ]

    markers = [
        ("circle", "o"),
        ("pixel", ","),
        ("triangle_down", "v"),
        ("triangle_up", "^"),
        ("triangle_left", "<"),
        ("triangle_right", ">"),
        ("square", "s"),
        ("pentagon", "p"),
        ("star", "*"),
        ("hexagon1", "h"),
        ("hexagon2", "H"),
        ("plus", "+"),
        ("x", "x"),
        ("diamond", "D"),
        ("thin_diamond", "d"),
    ]
    # Dictionary to store metrics from all files
    all_metrics = {}

    for i, file_path in enumerate(file_paths):
        # Load the file
        metrics = torch.load(file_path, map_location=torch.device("cpu"))
        metrics_list = []
        for metric in metrics:
            metrics_list.append(metric.detach().numpy())
        # Assuming the .pt files contain metrics such as loss and accuracy
        wd = file_path.split("wd")[1].split(".pt")[0]
        all_metrics[f"wd={wd}"] = metrics_list

    # Example: Plot the "loss" vs. "epoch" for each run
    plt.figure(figsize=(10, 6))

    for i in range(4):
        name = list(all_metrics.keys())[i]
        metrics = all_metrics[name]
        Xs = np.linspace(1, len(metrics) * 5, len(metrics))
        plt.plot(
            Xs[::5],
            metrics[::5],
            label=name,
            linestyle=line_styles[i][1],
            marker=markers[i][1],
            linewidth=3,
            markersize=5,
        )

    # Customize the plot
    # plt.title("Training Process for Different Configurations")
    plt.grid(True)
    # plt.yscale("log")
    # plt.title("ratio vs Epochs")
    plt.xlabel("Iterations", fontsize=24)
    plt.ylabel("val error", fontsize=24)
    plt.legend(fontsize=16)

    # Save the plot as an EPS file
    plt.savefig("LORA/val_err.eps", format="eps", dpi=100, bbox_inches="tight")

    # Optionally, display the plot
    plt.show()


def plot_attn_acc():
    # Read the CSV file
    df = pd.read_csv("attention/val_acc.csv")

    num_cols = len(df.columns)

    line_styles = [
        ("solid", "-"),
        ("dashed", "--"),
        ("dashdot", "-."),
        ("dotted", ":"),
        ("none", ""),
    ]

    markers = [
        ("circle", "o"),
        ("triangle_down", "v"),
        ("triangle_up", "^"),
        ("triangle_left", "<"),
        ("triangle_right", ">"),
        ("square", "s"),
        ("pentagon", "p"),
        ("star", "*"),
        ("hexagon1", "h"),
        ("hexagon2", "H"),
        ("plus", "+"),
        ("x", "x"),
        ("diamond", "D"),
        ("thin_diamond", "d"),
    ]

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot each column
    for idx in range(1, num_cols, 3):

        reg_value = df.columns[idx].split("-")[0].split("_")[-1]
        print(
            df.columns[idx],
            reg_value,
            line_styles[(idx // 3) // 5][1],
            markers[(idx // 3) // 14][1],
        )

        plt.plot(
            df.index[::5],
            (1 - df.iloc[::5, idx]) * 100.0,
            label=r"$\lambda=$" + reg_value,
            linestyle=line_styles[(idx // 3) // 5][1],
            marker=markers[idx // 3][1],
            linewidth=4,
            markersize=10,
        )

    # Customize the plot
    # plt.title("Training Process for Different Configurations")
    plt.grid(True)
    # plt.yscale("log")
    # Set the x-axis to start from 0
    plt.xlim(left=0)
    plt.xlabel("iterations", fontsize=24)
    plt.ylabel("val error", fontsize=24)
    plt.legend(fontsize=16, loc="best")

    # Save the plot as an EPS file
    plt.savefig("attention/val_error.eps", format="eps", dpi=100, bbox_inches="tight")

    # print("Plot saved as training_process.eps")

    # Optionally, display the plot
    plt.show()


def plot_attn_ratio():
    # Read the CSV file
    df = pd.read_csv("attention/nuc_by_frob_norm.csv")

    num_cols = len(df.columns)

    line_styles = [
        ("solid", "-"),
        ("dashed", "--"),
        ("dashdot", "-."),
        ("dotted", ":"),
        ("none", ""),
    ]

    markers = [
        ("circle", "o"),
        ("pixel", ","),
        ("triangle_down", "v"),
        ("triangle_up", "^"),
        ("triangle_left", "<"),
        ("triangle_right", ">"),
        ("square", "s"),
        ("pentagon", "p"),
        ("star", "*"),
        ("hexagon1", "h"),
        ("hexagon2", "H"),
        ("plus", "+"),
        ("x", "x"),
        ("diamond", "D"),
        ("thin_diamond", "d"),
    ]

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Plot each column
    for idx in range(1, num_cols, 3):

        reg_value = df.columns[idx].split("_")[3].split("-")[0]
        print(
            df.columns[idx],
            reg_value,
            line_styles[(idx // 3) // 5][1],
            markers[(idx // 3) // 15][1],
        )

        plt.plot(
            df.index[::5],
            df.iloc[::5, idx],
            label=r"$\lambda=$" + reg_value,
            linestyle=line_styles[(idx // 3) // 5][1],
            marker=markers[idx // 3][1],
            linewidth=4,
            markersize=10,
        )

    # Customize the plot
    # plt.title("Training Process for Different Configurations")
    plt.grid(True)
    plt.xlim(left=0)
    # plt.yscale("log")
    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Nuclear norm / Frobenius norm", fontsize=24)
    plt.legend(fontsize=16, loc="best", ncol=2)

    # Save the plot as an EPS file
    # plt.savefig(
    #     "attention/nuc_by_frob_norm.eps", format="eps", dpi=100, bbox_inches="tight"
    # )

    # print("Plot saved as training_process.eps")

    # Optionally, display the plot
    plt.show()


if __name__ == "__main__":
    plot_lora_ratio()
