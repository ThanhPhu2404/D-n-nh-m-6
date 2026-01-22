import cv2
import torch
from collections import Counter

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
COCO_CLASSES = model.names
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

def detect_vehicles_from_video(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)

    counter = Counter()
    frame_count = 0

    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        predictions = results.pred[0]
        labels = predictions[:, -1].cpu().numpy().astype(int)
        class_names = [COCO_CLASSES[i] for i in labels]

        for name in class_names:
            if name == 'motorcycle':
                counter['motorbike'] += 1
            elif name in VEHICLE_CLASSES:
                counter[name] += 1

        frame_count += 1

    cap.release()

    return {
        'car': counter.get('car', 0),
        'truck': counter.get('truck', 0),
        'bus': counter.get('bus', 0),
        'motorbike': counter.get('motorbike', 0)
    }
