#!/usr/bin/env python3
"""
SAM 3 (text-prompted) semantic masks for training frames.

    # 1) tune prompts on a few frames first: saves colour overlays, nothing else
    python sam3_masks.py --data data/ --laps 1 --cams fl fr r --preview 40

    # 2) full run (resumable: existing masks are skipped)
    python sam3_masks.py --data data/ --laps 0 1 2 3 4 5 6 7 8 --cams fl fr r

Output (under data/sam3/):
    <cam>/<stamp_ns>.png   uint8 label map at full image resolution, ids in CLASSES below
    stats.parquet          per frame: pixel fraction of every class
    preview/*.jpg          overlays (only with --preview)
    classes.json           id -> name, prompts, thresholds actually used

Needs: pip install -U transformers accelerate pillow ; the facebook/sam3 weights are gated
on Hugging Face: accept the licence on the model page, then `huggingface-cli login`.
"""
import argparse, json, time
import itertools
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from lap_index import load_frames

# id, name, prompts (union), score threshold.  Painted in this order: later classes win,
# so thin / important things (kerb, line, car) end up on top of road and background.
CLASSES = [
    (1, "sky",      ["night sky"],                              0.40),
    (2, "building", ["grandstand", "building"],                 0.40),
    (3, "barrier",  ["safety barrier", "fence", "concrete wall"], 0.35),
    (4, "road",     ["asphalt road", "race track"],             0.35),
    (5, "kerb",     ["red and white curb", "curb"],             0.35),
    (6, "line",     ["white line painted on the road"],         0.35),
    (7, "light",    ["floodlight"],                             0.45),
    (8, "car",      ["race car", "car"],                        0.45),
]
PALETTE = {0: (0, 0, 0), 1: (235, 206, 135), 2: (180, 105, 255), 3: (0, 165, 255),
           4: (128, 128, 128), 5: (0, 0, 255), 6: (255, 255, 255), 7: (0, 255, 255),
           8: (255, 0, 0)}  # BGR


def load_model(fp32):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if (fp32 or dev == "cpu") else torch.bfloat16
    try:
        model = Sam3Model.from_pretrained("facebook/sam3", dtype=dtype)
    except TypeError:                                  # transformers 4.x naming
        model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=dtype)
    model = model.to(dev).eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    return model, proc, dev, dtype


@torch.inference_mode()
def segment(model, proc, dev, dtype, pil_images, text_inputs, classes):
    """Returns a (B, H, W) uint8 label map for a batch of same-size images."""
    inp = proc(images=pil_images, return_tensors="pt").to(dev)
    sizes = inp["original_sizes"].tolist()
    vis = model.get_vision_features(pixel_values=inp["pixel_values"].to(dtype))
    B = len(pil_images)
    H, W = int(sizes[0][0]), int(sizes[0][1])
    label = np.zeros((B, H, W), np.uint8)
    for cid, _, prompts, thr in classes:
        for p in prompts:
            ti = text_inputs[p]
            out = model(vision_embeds=vis,
                        input_ids=ti["input_ids"].expand(B, -1),
                        attention_mask=ti["attention_mask"].expand(B, -1))
            res = proc.post_process_instance_segmentation(
                out, threshold=thr, mask_threshold=0.5, target_sizes=sizes)
            for b, r in enumerate(res):
                if len(r["masks"]):
                    label[b][r["masks"].bool().any(0).cpu().numpy()] = cid
    return label


def overlay(bgr, label):
    col = np.zeros_like(bgr)
    for cid, c in PALETTE.items():
        col[label == cid] = c
    out = cv2.addWeighted(bgr, 0.55, col, 0.45, 0)
    y = 20
    for cid, name, _, _ in CLASSES:
        cv2.putText(out, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PALETTE[cid], 2)
        y += 22
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--laps", nargs="+", type=int, required=True)
    ap.add_argument("--cams", nargs="+", default=["fl", "fr", "r"])
    ap.add_argument("--every", type=int, default=1, help="take every k-th frame per camera")
    ap.add_argument("--classes", nargs="+", default=None, help="subset of class names")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--preview", type=int, default=0, help="only N evenly spaced frames + overlays")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    data = Path(a.data)
    out = data / "sam3"
    classes = [c for c in CLASSES if a.classes is None or c[1] in a.classes]
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(
        [{"id": c[0], "name": c[1], "prompts": c[2], "threshold": c[3]} for c in classes], indent=2))

    f = load_frames(data, laps=a.laps, cams=a.cams, require_gt=False)
    f = pd.concat([g.iloc[::a.every] for _, g in f.groupby("cam")]).sort_values("stamp_ns")
    if a.preview:
        f = f.iloc[np.linspace(0, len(f) - 1, a.preview).astype(int)]
        (out / "preview").mkdir(exist_ok=True)
    else:
        done = f.apply(lambda r: (out / r.cam / f"{r.stamp_ns}.png").exists(), axis=1)
        print(f"{done.sum()} / {len(f)} already done")
        f = f[~done]
    f = f.reset_index(drop=True)
    for cam in a.cams:
        (out / cam).mkdir(exist_ok=True)

    model, proc, dev, dtype = load_model(a.fp32)
    text_inputs = {p: proc(text=p, return_tensors="pt").to(dev)
                   for c in classes for p in c[2]}

    def read(path):
        bgr = cv2.imread(str(data / path))
        return bgr, Image.fromarray(bgr[:, :, ::-1])

    stats_path = out / "stats.parquet"
    stats = [pd.read_parquet(stats_path)] if stats_path.exists() and not a.preview else []
    rows, t0 = [], time.time()
    with ThreadPoolExecutor(4) as pool:
        batches = [f.iloc[i:i + a.batch] for i in range(0, len(f), a.batch)]
        # bounded prefetch: Executor.map would submit ALL batches at once and load every
        # image of the run into RAM, which gets the process killed by the OOM killer
        load = lambda b: [read(p) for p in b.path]
        queue = deque()
        it = iter(batches)
        for b in itertools.islice(it, 3):
            queue.append((b, pool.submit(load, b)))
        k = -1
        while queue:
            b, fut = queue.popleft()
            nb = next(it, None)
            if nb is not None:
                queue.append((nb, pool.submit(load, nb)))
            imgs = fut.result()
            k += 1
            labels = segment(model, proc, dev, dtype, [im[1] for im in imgs], text_inputs, classes)
            for r, (bgr, _), lab in zip(b.itertuples(), imgs, labels):
                if a.preview:
                    cv2.imwrite(str(out / "preview" / f"{r.cam}_{r.stamp_ns}.jpg"), overlay(bgr, lab))
                    continue
                cv2.imwrite(str(out / r.cam / f"{r.stamp_ns}.png"), lab)
                frac = np.bincount(lab.ravel(), minlength=256) / lab.size
                rows.append({"stamp_ns": r.stamp_ns, "cam": r.cam, "lap": r.lap,
                             **{c[1]: frac[c[0]] for c in classes}})
            n = (k + 1) * a.batch
            if k % 25 == 0:
                rate = n / (time.time() - t0)
                print(f"{min(n, len(f))}/{len(f)}  {rate:.1f} img/s  "
                      f"eta {(len(f) - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)
            if rows and k % 500 == 0:
                pd.concat(stats + [pd.DataFrame(rows)]).to_parquet(stats_path)
    if rows:
        pd.concat(stats + [pd.DataFrame(rows)]).to_parquet(stats_path)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()