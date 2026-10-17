import csv
import sys
from collections import Counter

import pandas as pd
from mcap.reader import make_reader

bag_path = sys.argv[1]
laps_path = sys.argv[2]

laps = pd.read_csv(laps_path).sort_values("t_start")
counts = Counter()

image_types = (
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/Image",
    "sensor_msgs/CompressedImage",
)

with open(bag_path, "rb") as f, open("frames.csv", "w", newline="") as out:
    reader = make_reader(f)

    summary = reader.get_summary()
    if summary and summary.statistics:
        t0_ns = summary.statistics.message_start_time
    else:
        t0_ns = next(reader.iter_messages())[2].log_time
        f.seek(0)
        reader = make_reader(f)

    writer = csv.writer(out)
    writer.writerow([
        "frame_id", "camera", "timestamp_ns",
        "t_rel_s", "lap", "complete"
    ])

    frame_id = 0

    for schema, channel, message in reader.iter_messages():
        if schema is None or schema.name not in image_types:
            continue

        t_ns = message.log_time
        t_rel = (t_ns - t0_ns) / 1e9

        matched = laps[
            (laps.t_start <= t_rel) &
            (t_rel < laps.t_end)
        ]

        if matched.empty:
            continue

        row = matched.iloc[0]
        lap = int(row["lap"])

        writer.writerow([
            frame_id,
            channel.topic,
            t_ns,
            round(t_rel, 6),
            lap,
            bool(row["complete"])
        ])

        counts[(lap, channel.topic)] += 1
        frame_id += 1

print(f"Indexed {frame_id} frames")
for (lap, camera), count in sorted(counts.items()):
    print(f"Lap {lap}: {camera}: {count} frames")