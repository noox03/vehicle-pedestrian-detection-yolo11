"""Evaluate the model on a labelled validation set (YOLO format).

You need the dataset's data.yaml (paths to images and labels, class names
Vehicle and Pedestrian). Use the same dataset the model was trained on.

    python evaluate.py --data path/to/data.yaml
"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="val", choices=["val", "test"])
    args = p.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split)
    print(f"Precision   : {metrics.box.mp:.3f}")
    print(f"Recall      : {metrics.box.mr:.3f}")
    print(f"mAP50       : {metrics.box.map50:.3f}")
    print(f"mAP50-95    : {metrics.box.map:.3f}")
    for i, name in model.names.items():
        if i < len(metrics.box.ap50):
            print(f"  {name}: AP50 {metrics.box.ap50[i]:.3f}  AP50-95 {metrics.box.ap[i]:.3f}")


if __name__ == "__main__":
    main()
