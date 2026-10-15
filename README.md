# Vehicle and Pedestrian Detection with YOLO11n (foggy road scenes)

A small object detector for two classes, **Vehicle** and **Pedestrian**, built for driver assistance (ADAS) style scenes. It is tested here on foggy road images, where objects have low contrast and distant cars are hard to see.

![Foggy road input](assets/foggy_input.jpg)
![Detections](assets/foggy_detection.jpg)

*Left: input image. Right: model output (confidence threshold 0.25). Nearby vehicles are found with confidence around 0.9, and several distant, low contrast vehicles are also found.*

## Model

| Item | Value |
|---|---|
| Architecture | YOLO11n (nano), Ultralytics 8.3.241 |
| Starting point | pretrained `yolo11n.pt`, fine tuned |
| Classes | Vehicle, Pedestrian |
| Training | 150 epochs, image size 640, batch size 16, on Kaggle |
| Weights | `weights/best.pt` (about 5 MB) |

## Results

Validation metrics stored in the checkpoint at the best epoch:

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.829 | 0.767 | 0.813 | 0.527 |

Reproduce them with `evaluate.py` (see below). If your numbers differ, trust your own run.

Speed can be measured on your own hardware with `benchmark.py` (see Usage).

## Dataset

The training data was provided by my school (ENSAM Casablanca) as part of my Master's studies and was used through Kaggle. The images and labels are **not included** in this repository, because I do not own the dataset and cannot redistribute it. The model was trained on two classes, Vehicle and Pedestrian.

## Method

I fine tuned the pretrained `yolo11n.pt` model from Ultralytics on the two class dataset for 150 epochs at image size 640 with batch size 16, training on Kaggle. I kept the checkpoint with the best validation score (`weights/best.pt`) and evaluated it with precision, recall, mAP50 and mAP50 to 95. I then tested it on foggy road images, where small distant objects get lower confidence.

## Usage

```bash
pip install -r requirements.txt

# one image (saves the result in results/)
python detect.py --source assets/foggy_input.jpg

# a folder of images, or a video
python detect.py --source my_images/
python detect.py --source road.mp4

# speed measurement
python benchmark.py --source assets/foggy_input.jpg --runs 100

# accuracy on a labelled validation set (YOLO format data.yaml)
python evaluate.py --data path/to/data.yaml
```

## Limitations

- Only two classes, and only trained on the data described above.
- Small, distant objects in heavy fog get lower confidence, and crowded scenes can produce overlapping boxes.
- This is a research and learning prototype. It is not a safety certified system and must not be used to make driving decisions.

## Licence

The model and training code use Ultralytics YOLO, which is licensed under AGPL-3.0 (https://ultralytics.com/license). Please keep that in mind if you reuse or redistribute the weights or scripts.

## Author

Houssam Chakour, Master's student, FH Upper Austria. Contact: houssamchakour56@gmail.com
