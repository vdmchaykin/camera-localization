import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("gt.csv")
df = df.sort_values("timestamp_ns")

t = (df["timestamp_ns"] - df["timestamp_ns"].iloc[0]) / 1e9 / 60

plt.figure(figsize=(12, 9))
plt.scatter(df["x"], df["y"], c=t, cmap="viridis", s=1)
plt.colorbar(label="Time (minutes)")
plt.scatter(df["x"].iloc[0], df["y"].iloc[0], c="red", s=80, label="Start")
plt.scatter(df["x"].iloc[-1], df["y"].iloc[-1], c="orange", s=80, label="End")
plt.axis("equal")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Robot Trajectory over Time")
plt.grid(True)
plt.legend()
plt.savefig("gt_trajectory_time.png", dpi=200)