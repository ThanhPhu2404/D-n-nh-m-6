import torch
import cv2

def load_model():
    model = torch.hub.load('../yolov5', 'custom', path='../yolov5/yolov5s.pt', source='local')
    model.conf = 0.4
    return model

def detect_vehicles_live(video_path):
    model = load_model()
    cap = cv2.VideoCapture(video_path)

    vehicle_classes = ['car', 'truck', 'bus', 'motorbike']
    vehicle_count = {cls: 0 for cls in vehicle_classes}

    ret, frame = cap.read()
    if ret:
        results = model(frame)
        for *box, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label in vehicle_classes:
                vehicle_count[label] += 1

    cap.release()
    return vehicle_count

# yolov5s.pt dùng để detect các lớp COCO (car, bus, truck, motorbike…) Được huấn luyện trên bộ dữ liệu MS COCO. Bộ COCO có 80 lớp đối tượng phổ biến trong đời sống.