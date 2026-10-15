"""Run the trained YOLO11n model on an image, a folder of images, or a video.

Examples:
    python detect.py --source assets/foggy_input.jpg
    python detect.py --source my_images/ --conf 0.25
    python detect.py --source road.mp4 --save-dir results
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def annotate_image(model, path, conf, imgsz, save_dir, show):
    start = time.perf_counter()
    result = model.predict(source=str(path), conf=conf, imgsz=imgsz, verbose=False)[0]
    ms = (time.perf_counter() - start) * 1000
    annotated = result.plot()
    out = save_dir / f"{Path(path).stem}_detected.jpg"
    cv2.imwrite(str(out), annotated)
    counts = {}
    for c in result.boxes.cls.tolist():
        name = model.names[int(c)]
        counts[name] = counts.get(name, 0) + 1
    print(f"{Path(path).name}: {counts or 'no detections'} in {ms:.1f} ms -> {out}")
    if show:
        cv2.imshow("Detection (press any key)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def annotate_video(model, path, conf, imgsz, save_dir):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = save_dir / f"{Path(path).stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frames, total_ms = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        start = time.perf_counter()
        result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        total_ms += (time.perf_counter() - start) * 1000
        writer.write(result.plot())
        frames += 1
    cap.release()
    writer.release()
    if frames:
        print(f"{frames} frames, mean {total_ms / frames:.1f} ms per frame -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--source", required=True, help="image, folder of images, or video")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save-dir", default="results")
    p.add_argument("--show", action="store_true", help="open a window for each image")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    if source.is_dir():
        files = sorted(f for f in source.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise SystemExit("No images found in that folder.")
        for f in files:
            annotate_image(model, f, args.conf, args.imgsz, save_dir, args.show)
    elif source.suffix.lower() in VIDEO_EXT:
        annotate_video(model, source, args.conf, args.imgsz, save_dir)
    else:
        annotate_image(model, source, args.conf, args.imgsz, save_dir, args.show)


if __name__ == "__main__":
    main()
