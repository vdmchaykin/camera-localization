#!/usr/bin/env python3
"""All camera frames -> frames/<cam>/<stamp_ns>.jpg at half resolution (753x364), one pass over the mcap.

    python3 extract_frames.py data/c0_.../c0_..._0.mcap frames

Half-resolution intrinsics: fx,fy *= 0.5; cx' = (cx + 0.5) * 0.5 - 0.5 (same for cy). See camera_model.py.
"""
import os, sys, time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

CAMS = ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']
SCALE = 0.5
QUALITY = 92


def work(path, data):
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    im = cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])


def main(bag, out):
    for c in CAMS:
        os.makedirs(os.path.join(out, c), exist_ok=True)
    topics = [f'/constructor0/sensor/camera_{c}/compressed_image' for c in CAMS]
    t0, n = time.time(), 0
    with open(bag, 'rb') as f, ThreadPoolExecutor(10) as ex:
        futs = []
        for _, ch, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=topics, log_time_order=False):
            cam = ch.topic.split('/')[-2].replace('camera_', '')
            st = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            p = os.path.join(out, cam, f'{st}.jpg')
            if not os.path.exists(p):
                futs.append(ex.submit(work, p, bytes(msg.data)))
            n += 1
            if len(futs) > 200:
                for fu in futs:
                    fu.result()
                futs = []
            if n % 5000 == 0:
                print(f'{n} frames, {time.time() - t0:.0f} s', flush=True)
        for fu in futs:
            fu.result()
    print(f'done {n} frames in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
