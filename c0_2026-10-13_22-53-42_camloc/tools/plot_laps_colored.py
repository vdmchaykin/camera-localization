import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sync_frames.csv")
df = df.sort_values("timestamp_ns")

fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.get_cmap("tab10")

for lap, group in df.groupby("lap"):
    group = group.sort_values("timestamp_ns")
    ax.plot(
        group["x"], group["y"],
        color=colors(int(lap) % 10),
        linewidth=1.8,
        label=f"Lap {lap}"
    )
    ax.scatter(
        group["x"].iloc[0], group["y"].iloc[0],
        color=colors(int(lap) % 10),
        s=35
    )

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Camera Localization: Trajectories by Lap")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig("sync_laps_colored.png", dpi=200)
print("Saved: sync_laps_colored.png")