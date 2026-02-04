import os
import cv2
import numpy as np
import time
import torch
from .sort import Sort
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
from .models import Node, Edge, Vehicle
from .services import dijkstra
from django.core.files.storage import FileSystemStorage

# ===== CẤU HÌNH HỆ THỐNG =====

# 1. ĐỘ PHÂN GIẢI AI (Giữ 480 để cân bằng giữa tốc độ và chính xác)
MODEL_IMGSZ = 480 

# 2. KHOẢNG CÁCH THỰC 
REAL_DISTANCE = 10.0 

# Kích thước hiển thị cửa sổ
DISPLAY_W, DISPLAY_H = 1280, 720 
LINE_START_Y = 350
LINE_END_Y = 550

# ===== KHỞI TẠO MODEL =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- DEVICE: {DEVICE.upper()} | FRAME_SKIP: OFF ---")

try:
    model_path = os.path.join(settings.BASE_DIR, 'yolov8n.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(settings.BASE_DIR, 'yolov5su.pt')
    model = YOLO(model_path)
    model.to(DEVICE) 
except Exception as e:
    print(f"Lỗi load model: {e}. Đang tải model mặc định...")
    model = YOLO('yolov8n.pt')

tracker = Sort(max_age=50, min_hits=2, iou_threshold=0.3)

# ====== CÁC VIEW CƠ BẢN (GIỮ NGUYÊN) ======
def contact_view(request): return render(request, 'contact.html')
def home(request): return render(request, 'index.html')
def about_view(request):
    vehicles = Vehicle.objects.all()
    return render(request, 'about.html', {'vehicles': vehicles, 'total_vehicles': vehicles.count()})
def joblist_view(request): return render(request, 'job-list.html')
def jobdetail_view(request): return render(request, 'job-detail.html')
def category_view(request): return render(request, 'category.html')
def testimonial_view(request): return render(request, 'testimonial.html')
def error_404_view(request, exception=None): return render(request, '404.html')

def shortest_path(request):
    try:
        start_id = int(request.GET.get('start_id'))
        end_id = int(request.GET.get('end_id'))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'Tham số không hợp lệ'}, status=400)
    nodes = Node.objects.all()
    edges = Edge.objects.all()
    path_node_ids = dijkstra(nodes, edges, start_id, end_id)
    path_coords = []
    if path_node_ids:
        nodes_dict = {node.id: node for node in nodes}
        for nid in path_node_ids:
            node = nodes_dict.get(nid)
            if node: path_coords.append({'lat': node.lat, 'lng': node.lng})
    return JsonResponse({'path': path_coords})

# ====== CÁC HÀM HỖ TRỢ ======
def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def refine_vehicle_class(class_name, x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    area = w * h
    if class_name == 'motorcycle': return 'motorbike'
    if class_name in ['truck', 'bus']:
        if h > 0.6 * w and area > 45000: return 'bus'
        else: return 'truck'
    return class_name

def get_stable_class(track_id, track_class_votes):
    votes = track_class_votes.get(track_id, {})
    if not votes: return None
    return max(votes, key=votes.get)

# ====== VIEW ĐẾM XE THƯỜNG ======
@csrf_exempt
def detect_vehicles_view(request):
    if request.method != 'POST': pass
    video_path = os.path.join(settings.BASE_DIR, 'main', 'static', 'img', 'road.mp4')
    cap = cv2.VideoCapture(video_path)
    WINDOW_NAME = "Dem xe"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    
    counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorbike': 0}
    counted_ids = set()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (1280, 720))
        results = model(frame, verbose=False, device=DEVICE, imgsz=MODEL_IMGSZ)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            raw_cls = model.names[cls_id]
            if raw_cls in ['car', 'truck', 'bus', 'motorcycle', 'motorbike']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])
        
        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(dets_np)
        
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
    return JsonResponse({'counts': counts})

# ====== VIEW ĐO TỐC ĐỘ (FULL FRAMES) ======
@csrf_exempt
def detect_vehicles_speed_view(request):
    if request.method != 'POST': pass
    video_path = os.path.join(settings.BASE_DIR, 'main', 'static', 'img', 'videotransport1.mp4')
    if not os.path.exists(video_path): return JsonResponse({'error': 'Không tìm thấy video'}, status=404)
    return process_video_speed(video_path)

@csrf_exempt
def detect_vehicles_speed_upload_view(request):
    if request.method != 'POST': return JsonResponse({'error': 'Yêu cầu POST'}, status=400)
    if 'video' not in request.FILES: return JsonResponse({'error': 'Chưa chọn file'}, status=400)
    uploaded_file = request.FILES['video']
    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
    filename = fs.save(uploaded_file.name, uploaded_file)
    video_path = fs.path(filename)
    try:
        return process_video_speed(video_path)
    finally:
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass

def process_video_speed(video_path):
    cap = cv2.VideoCapture(video_path)
    
    WINDOW_NAME = "Giam sat Giao thong - Full Frames"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    colors = {'car': (0, 255, 0), 'truck': (255, 0, 0), 'bus': (0, 255, 255), 'motorbike': (0, 0, 255)}
    counts = {k: 0 for k in colors}
    vehicle_speeds = []
    valid_speeds_history = [] 

    tracker = Sort(max_age=50, min_hits=2, iou_threshold=0.3)
    MIN_TIME_PASS = 0.2 

    track_state = {}
    track_class_votes = {}
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Lấy thời gian thực từ video (không phụ thuộc vào tốc độ xử lý của máy)
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        
        # Nhận diện AI
        results = model(frame, verbose=False, stream=True, conf=0.15, device=DEVICE, imgsz=MODEL_IMGSZ)
        
        detections = []
        current_classes = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                raw_cls = model.names[cls_id]
                if raw_cls in ['car', 'truck', 'bus', 'motorcycle', 'motorbike']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    final_cls = refine_vehicle_class(raw_cls, x1, y1, x2, y2)
                    detections.append([x1, y1, x2, y2, conf])
                    current_classes.append(final_cls)

        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(dets_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            reference_y = y2 
            
            # Gán loại xe cho tracker
            best_cls = None
            max_iou = 0
            for i, det in enumerate(detections):
                iou = bbox_iou(track[:4], det[:4])
                if iou > 0.4 and iou > max_iou:
                    max_iou = iou
                    best_cls = current_classes[i]
            
            if best_cls:
                track_class_votes.setdefault(track_id, {})
                track_class_votes[track_id][best_cls] = track_class_votes[track_id].get(best_cls, 0) + 1
            
            stable_class = get_stable_class(track_id, track_class_votes)
            if not stable_class: continue

            track_state.setdefault(track_id, {'start_time': None, 'done': False, 'speed': 0})
            state = track_state[track_id]

            # Logic đo tốc độ
            if state['start_time'] is None and reference_y >= LINE_START_Y and reference_y < LINE_END_Y:
                state['start_time'] = current_time_sec
            
            elif state['start_time'] is not None and not state['done'] and reference_y >= LINE_END_Y:
                time_diff = current_time_sec - state['start_time']
                
                if time_diff > MIN_TIME_PASS: 
                    speed_ms = REAL_DISTANCE / time_diff
                    raw_speed_kmh = speed_ms * 3.6
                    
                    # Bộ lọc làm mượt vận tốc
                    final_speed = 0.0
                    if len(valid_speeds_history) == 0:
                        if 3 < raw_speed_kmh < 150: final_speed = raw_speed_kmh
                        else: final_speed = 20.0
                    else:
                        avg_speed = sum(valid_speeds_history) / len(valid_speeds_history)
                        if abs(raw_speed_kmh - avg_speed) <= 20.0: final_speed = raw_speed_kmh
                        else: final_speed = avg_speed

                    if final_speed > 0:
                        state['speed'] = final_speed
                        state['done'] = True
                        counts[stable_class] += 1
                        vehicle_speeds.append({
                            'id': int(track_id),
                            'type': stable_class,
                            'speed': round(final_speed, 1)
                        })
                        valid_speeds_history.append(final_speed)
                        if len(valid_speeds_history) > 30: valid_speeds_history.pop(0)

            # Vẽ Box và Label
            color = colors.get(stable_class, (255, 255, 255))
            label = f"{stable_class} {track_id}"
            if state['done']:
                label += f" {state['speed']:.0f}km/h"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Vẽ vạch đo
        cv2.line(frame, (0, LINE_START_Y), (DISPLAY_W, LINE_START_Y), (255, 0, 0), 2)
        cv2.line(frame, (0, LINE_END_Y), (DISPLAY_W, LINE_END_Y), (0, 0, 255), 2)
        cv2.putText(frame, f"Total Vehicles: {sum(counts.values())}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p') or key == ord(' '): cv2.waitKey(-1)

    cap.release()
    cv2.destroyAllWindows()

    return JsonResponse({
        'counts': counts,
        'total_vehicles': sum(counts.values()),
        'vehicles': vehicle_speeds
    })