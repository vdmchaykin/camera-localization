#!/usr/bin/env python3
"""
SAM 3 in VIDEO mode: masks that are consistent over time (detector + memory tracker).

Same classes / prompts / output format as sam3_masks.py, written to data/sam3_video/.

    # 1) look first: one chunk of one camera -> overlay video data/sam3_video/preview_<cam>.mp4
    python sam3_video_masks.py --data data/ --laps 1 --cams cl --preview

    # 2) all training laps, all cameras (resumable, run with nohup python -u ...)
    python sam3_video_masks.py --data data/ --laps 0 1 2 4 5 7 8 --cams cl cr fl fr r rl rr

How it works: frames of one camera are cut into chunks of --chunk frames (10 Hz -> 20 s by
default) and every chunk is one SAM 3 video session with all prompts. Inside a chunk masks
are tracked with memory, so they do not flicker; a new chunk restarts tracking (object ids
are not kept, only class labels are saved, so that does not matter).
This is offline labelling of training frames: the tracker may look at the whole chunk
(not causal). Never use it this way at inference.
"""
import argparse, json, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

from lap_index import load_frames
from sam3_masks import CLASSES, overlay


def load_model(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: no CUDA, this will be extremely slow")
    dtype = torch.float32 if (a.fp32 or dev == "cpu") else torch.bfloat16
    cfg = Sam3VideoConfig.from_pretrained("facebook/sam3")
    cfg.score_threshold_detection = a.det_thr      # keep detections above this
    cfg.new_det_thresh = a.new_thr                 # start a new track above this
    model = Sam3VideoModel.from_pretrained("facebook/sam3", config=cfg).to(dev, dtype=dtype).eval()
    proc = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    return model, proc, dev, dtype


def chunks_of(f, size, max_gap_s=0.5):
    """Consecutive frames of one camera, split at lap changes and time gaps."""
    f = f.sort_values("stamp_ns").reset_index(drop=True)
    brk = (f.lap.diff().fillna(0) != 0) | (f.stamp_ns.diff().fillna(0) * 1e-9 > max_gap_s)
    for _, seg in f.groupby(brk.cumsum()):
        for i in range(0, len(seg), size):
            yield seg.iloc[i:i + size]


@torch.inference_mode()
def segment_chunk(model, proc, dev, dtype, frames, prompt_class):
    sess = proc.init_video_session(video=frames, inference_device=dev,
                                   processing_device="cpu", video_storage_device="cpu",
                                   dtype=dtype)
    proc.add_text_prompt(sess, list(prompt_class))
    H, W = frames[0].shape[:2]
    labels = np.zeros((len(frames), H, W), np.uint8)
    order = {cid: i for i, (cid, *_rest) in enumerate(CLASSES)}   # paint order
    for out in model.propagate_in_video_iterator(inference_session=sess,
                                                 max_frame_num_to_track=len(frames)):
        res = proc.postprocess_outputs(sess, out)
        obj_cls = {}
        for prompt, ids in res["prompt_to_obj_ids"].items():
            for oid in ids:
                obj_cls[int(oid)] = prompt_class[prompt]
        masks = res["masks"].bool().cpu().numpy()
        oids = [int(o) for o in res["object_ids"]]
        lab = labels[out.frame_idx]
        for j in sorted(range(len(oids)), key=lambda j: order.get(obj_cls.get(oids[j], 0), -1)):
            cid = obj_cls.get(oids[j])
            if cid is not None:
                lab[masks[j]] = cid
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--laps", nargs="+", type=int, required=True)
    ap.add_argument("--cams", nargs="+", default=["cl", "cr", "fl", "fr", "r", "rl", "rr"])
    ap.add_argument("--classes", nargs="+", default=None, help="subset of class names")
    ap.add_argument("--chunk", type=int, default=200, help="frames per video session")
    ap.add_argument("--det-thr", type=float, default=0.4, help="detection score threshold")
    ap.add_argument("--new-thr", type=float, default=0.5, help="score to start a new track")
    ap.add_argument("--preview", action="store_true", help="first chunk only + overlay mp4")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    data = Path(a.data)
    out = data / "sam3_video"
    out.mkdir(parents=True, exist_ok=True)
    classes = [c for c in CLASSES if a.classes is None or c[1] in a.classes]
    prompt_class = {p: cid for cid, _, prompts, _ in classes for p in prompts}
    (out / "classes.json").write_text(json.dumps(
        [{"id": c[0], "name": c[1], "prompts": c[2]} for c in classes], indent=2))

    model, proc, dev, dtype = load_model(a)
    frames_all = load_frames(data, laps=a.laps, cams=a.cams, require_gt=False)
    total = len(frames_all)
    done_n, t0 = 0, time.time()

    for cam in a.cams:
        (out / cam).mkdir(exist_ok=True)
        f = frames_all[frames_all.cam == cam]
        for ch in chunks_of(f, a.chunk):
            paths = [out / cam / f"{s}.png" for s in ch.stamp_ns]
            if not a.preview and all(p.exists() for p in paths):
                done_n += len(ch)
                continue
            imgs = [cv2.imread(str(data / p))[:, :, ::-1].copy() for p in ch.path]
            labels = segment_chunk(model, proc, dev, dtype, imgs, prompt_class)
            if a.preview:
                vp = out / f"preview_{cam}.mp4"
                h, w = labels.shape[1:]
                vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                for im, lab in zip(imgs, labels):
                    vw.write(overlay(np.ascontiguousarray(im[:, :, ::-1]), lab))
                vw.release()
                print(f"saved {vp} ({len(imgs)} frames, lap {ch.lap.iloc[0]})")
                break
            for p, lab in zip(paths, labels):
                cv2.imwrite(str(p), lab)
            done_n += len(ch)
            rate = done_n / (time.time() - t0)
            print(f"[{cam}] lap {ch.lap.iloc[0]}: {done_n}/{total} frames  {rate:.1f} img/s  "
                  f"eta {(total - done_n) / max(rate, 1e-6) / 60:.0f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()