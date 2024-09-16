import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("recon_MSE.csv")

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
plt.yscale("log")
plt.xlabel("iterations", fontsize=16)
plt.ylabel("reconstruction MSE (log scale)", fontsize=16)
plt.legend(fontsize=16)


# Save the plot as an EPS file
plt.savefig("recon_MSE.eps", format="eps", dpi=100, bbox_inches="tight")

print("Plot saved as training_process.eps")

# Optionally, display the plot
plt.show()