"""Measure inference latency of the model on one image.

Run it on the machine you want to report numbers for, and write the real
output into the README (do not copy numbers from another machine).

    python benchmark.py --source assets/foggy_input.jpg --runs 100
    python benchmark.py --source assets/foggy_input.jpg --device cpu
"""
import argparse
import statistics
import time

import cv2
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--source", required=True, help="path to one image")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cpu, 0 (first GPU), ...")
    args = p.parse_args()

    image = cv2.imread(args.source)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.source}")
    model = YOLO(args.weights)

    kwargs = dict(imgsz=args.imgsz, verbose=False)
    if args.device is not None:
        kwargs["device"] = args.device

    for _ in range(args.warmup):
        model.predict(source=image, **kwargs)

    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        model.predict(source=image, **kwargs)
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    mean = statistics.mean(times)
    print(f"Runs: {args.runs} (after {args.warmup} warmup)")
    print(f"Mean latency  : {mean:.1f} ms  ({1000 / mean:.1f} FPS)")
    print(f"Median latency: {statistics.median(times):.1f} ms")
    print(f"95th percentile: {times[int(0.95 * len(times)) - 1]:.1f} ms")
    print("Note: includes pre and post processing, not only the network.")


if __name__ == "__main__":
    main()
