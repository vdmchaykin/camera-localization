#!/usr/bin/env python3
"""Night -> day copy of the training run with Qwen-Image-Edit-2511 + the lightx2v Lightning 4-step LoRA:
    python qwen_day.py TRAIN.mcap DAY.mcap

edits every camera image of TRAIN.mcap on one GPU into DAY_images/<cam>/<stamp_ns>.jpg (resumable: a rerun skips the
images already there), then writes DAY.mcap: TRAIN.mcap with every camera JPEG swapped for its day version, all other
bytes, stamps and topics unchanged.

Environment: Python 3.10.20, torch 2.10.0+cu126, diffusers 0.38.0, transformers 5.4.0, peft 0.18.1,
accelerate 1.13.0, safetensors 0.8.0, huggingface_hub 1.8.0, numpy 2.2.5, opencv-python-headless 4.13.0.92, pillow,
mcap. Both models are Apache-2.0 and not gated.
"""
import argparse, math, os, struct, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np

MODEL = ('Qwen/Qwen-Image-Edit-2511', '6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9')        # repo, revision used
LORA = ('lightx2v/Qwen-Image-Edit-2511-Lightning', '107d238d68b8387e32843d0b4a2e24ceb0d3ad86',
        'Qwen-Image-Edit-2511-Lightning-{}steps-V1.0-bf16.safetensors')
SIZE = (1504, 736)        # generation size: multiple of 32, ~native (resized back on write)
W, H = 1506, 728          # camera image size
SEED = 1234               # one seed for every frame: as consistent a "day" as per-frame editing allows
JPEG_QUALITY = 90         # matches the quantisation tables of the original camera JPEGs
PROMPT = ('Relight this photo as a bright, sunny daytime photo. Replace the black night sky with a clear blue sky, '
          'switch off all floodlights and street lamps and remove their glare, halos and the haze, and light the '
          'scene with natural sunlight and realistic soft shadows. Keep everything else exactly the same: the camera '
          'viewpoint and framing, and every object with its position, shape, colour and text. Do not add, remove or '
          'move any object. Photorealistic photo.')   # naming the car body made it paint one into the rear cameras


# ---------------------------------------------------------------- sensor_msgs/CompressedImage payload (CDR)
def jpeg_field(buf):
    """Offset of the uint32 length of `data`, the JPEG length, the trailing padding (some messages are padded to
    4 bytes after the JPEG) and the header stamp (ns)."""
    assert buf[1] == 1, 'expects little-endian CDR'
    sec, nsec = struct.unpack_from('<iI', buf, 4)
    o = 12
    for _ in range(2):                                     # header.frame_id, then format: string = len + bytes
        o += 4 + struct.unpack_from('<I', buf, o)[0]
        o = 4 + ((o - 4 + 3) & ~3)                         # CDR alignment counts from after the 4-byte header
    n = struct.unpack_from('<I', buf, o)[0]
    pad = len(buf) - o - 4 - n
    assert 0 <= pad <= 3, f'data must be the last field (pad {pad})'
    return o, n, pad, sec * 10**9 + nsec


def swap_jpeg(buf, jpg):
    o, _, pad, _ = jpeg_field(buf)
    out = buf[:o] + struct.pack('<I', len(jpg)) + jpg
    return out + b'\0' * ((-(len(out) - 4)) % 4 if pad else 0)   # same padding rule as the original message


def camera(topic):
    return topic.split('/')[3].removeprefix('camera_')     # /constructor0/sensor/camera_fl/compressed_image -> fl


def camera_images(path):
    """(cam, header stamp ns, JPEG bytes) of every camera image of the mcap, in file order."""
    from mcap.reader import make_reader
    with open(path, 'rb') as f:
        for _, ch, m in make_reader(f).iter_messages(log_time_order=False):
            if ch.topic.endswith('/compressed_image'):
                o, n, _, stamp = jpeg_field(m.data)
                yield camera(ch.topic), stamp, m.data[o + 4:o + 4 + n]


# ---------------------------------------------------------------- 1. generate
def load(steps, compile_):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    sched = FlowMatchEulerDiscreteScheduler.from_config({        # the Lightning distillation's scheduler (shift 3)
        'base_image_seq_len': 256, 'base_shift': math.log(3), 'invert_sigmas': False, 'max_image_seq_len': 8192,
        'max_shift': math.log(3), 'num_train_timesteps': 1000, 'shift': 1.0, 'shift_terminal': None,
        'stochastic_sampling': False, 'time_shift_type': 'exponential', 'use_beta_sigmas': False,
        'use_dynamic_shifting': True, 'use_exponential_sigmas': False, 'use_karras_sigmas': False})
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL[0], revision=MODEL[1], torch_dtype=torch.bfloat16,
                                                     scheduler=sched).to('cuda')
    pipe.load_lora_weights(LORA[0], revision=LORA[1], weight_name=LORA[2].format(steps))
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.set_progress_bar_config(disable=True)
    if compile_:                                  # same math, fused kernels; one fixed shape, so one compile
        pipe.transformer = torch.compile(pipe.transformer)
    return pipe


def edit(pipe, bgr, steps):
    import torch
    from PIL import Image
    img = Image.fromarray(cv2.resize(bgr, SIZE, interpolation=cv2.INTER_CUBIC)[:, :, ::-1])
    with torch.no_grad():                         # Lightning: no CFG (true_cfg_scale 1)
        out = pipe(image=[img], prompt=PROMPT, num_inference_steps=steps, width=SIZE[0], height=SIZE[1],
                   generator=torch.Generator('cuda').manual_seed(SEED), true_cfg_scale=1.0).images[0]
    return np.ascontiguousarray(np.array(out)[:, :, ::-1])


def write(path, bgr):
    """Atomic write at the camera size, so a killed job never leaves a truncated JPEG behind."""
    if bgr.shape[:2] != (H, W):
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA if bgr.shape[1] > W else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok, path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def generate(mcap, images, steps, compile_):
    pipe = load(steps, compile_)
    io, futs, tic = ThreadPoolExecutor(2), [], time.time()
    for cam, stamp, jpg in camera_images(mcap):
        p = images / cam / f'{stamp}.jpg'
        if p.exists():
            continue
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        futs.append(io.submit(write, p, edit(pipe, bgr, steps)))
        if len(futs) % 100 == 0:
            print(f'{len(futs)} images, {time.time() - tic:.0f} s', flush=True)
    for f in futs:
        f.result()                                # surface write errors
    print(f'done, {len(futs)} images in {time.time() - tic:.0f} s', flush=True)


# ---------------------------------------------------------------- 2. pack
def pack(mcap, images, out):
    from mcap.reader import make_reader
    from mcap.writer import CompressionType, Writer
    tmp = out + '.tmp'
    with open(mcap, 'rb') as fi, open(tmp, 'wb') as fo:
        r = make_reader(fi)
        h = r.get_header()
        w = Writer(fo, chunk_size=8 * 1024 * 1024, compression=CompressionType.ZSTD)
        w.start(profile=h.profile, library=h.library)
        schemas, channels, swapped = {}, {}, 0
        for schema, ch, msg in r.iter_messages(log_time_order=False):
            if schema.id not in schemas:
                schemas[schema.id] = w.register_schema(schema.name, schema.encoding, schema.data)
            if ch.id not in channels:
                channels[ch.id] = w.register_channel(ch.topic, ch.message_encoding, schemas[schema.id], ch.metadata)
            data = msg.data
            if ch.topic.endswith('/compressed_image'):
                stamp = jpeg_field(data)[3]
                jpg = (images / camera(ch.topic) / f'{stamp}.jpg').read_bytes()   # a missing day image stops here
                data = swap_jpeg(data, jpg)
                assert jpeg_field(data)[1::2] == (len(jpg), stamp), (ch.topic, stamp)
                swapped += 1
            w.add_message(channels[ch.id], msg.log_time, data, msg.publish_time, msg.sequence)
        w.finish()
    os.replace(tmp, out)
    print(f'{out}: {swapped} camera images swapped, all other messages unchanged')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mcap', help='night training run')
    ap.add_argument('out', help='day copy to write (DAY.mcap)')
    ap.add_argument('--steps', type=int, choices=[4, 8], default=4, help='Lightning LoRA steps (the run used 4)')
    ap.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True, help='torch.compile the DiT')
    a = ap.parse_args()
    images = Path(a.out).with_name(Path(a.out).stem + '_images')       # DAY.mcap -> DAY_images/
    generate(a.mcap, images, a.steps, a.compile)
    pack(a.mcap, images, a.out)


if __name__ == '__main__':
    main()
