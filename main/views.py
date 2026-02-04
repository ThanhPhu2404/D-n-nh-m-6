import os
import cv2
import numpy as np
import time
from .sort import Sort
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.gzip import gzip_page
from ultralytics import YOLO
from .models import Node, Edge, Vehicle
from .services import dijkstra
from django.core.files.storage import FileSystemStorage

# ===== CẤU HÌNH HIỂN THỊ & ĐO ĐẠC =====
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Vạch ảo để tính tốc độ (Pixel trên khung hình 1280x720)
LINE_START_Y = 350
LINE_END_Y = 550

# Khoảng cách thực tế giữa 2 vạch (Mét) - Cần đo thực địa để chính xác nhất
REAL_DISTANCE = 20.0 

# ===== KHỞI TẠO MODEL (Load 1 lần để tối ưu) =====
try:
    # Ưu tiên dùng model YOLOv8n (nhanh, nhẹ)
    model_path = os.path.join(settings.BASE_DIR, 'yolov8n.pt')
    if not os.path.exists(model_path):
        # Fallback về model cũ của bạn nếu chưa có v8
        model_path = os.path.join(settings.BASE_DIR, 'yolov5su.pt')
    
    model = YOLO(model_path)
except Exception as e:
    print(f"Lỗi load model: {e}. Đang tải model mặc định...")
    model = YOLO('yolov8n.pt')

# Khởi tạo tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)


# ====== VIEW TRANG (GIỮ NGUYÊN) ======
def contact_view(request):
    return render(request, 'contact.html')


def home(request):
    return render(request, 'index.html')


def about_view(request):
    vehicles = Vehicle.objects.all()
    total_vehicles = vehicles.count()
    return render(request, 'about.html', {
        'vehicles': vehicles,
        'total_vehicles': total_vehicles
    })


def joblist_view(request):
    return render(request, 'job-list.html')


def jobdetail_view(request):
    return render(request, 'job-detail.html')


def category_view(request):
    return render(request, 'category.html')


def testimonial_view(request):
    return render(request, 'testimonial.html')


def error_404_view(request, exception=None):
    return render(request, '404.html')


# ====== XỬ LÝ TÌM ĐƯỜNG NGẮN NHẤT (GIỮ NGUYÊN) ======
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
            if node:
                path_coords.append({'lat': node.lat, 'lng': node.lng})

    return JsonResponse({'path': path_coords})


# ====== CÁC HÀM HỖ TRỢ XỬ LÝ ẢNH ======

def bbox_iou(boxA, boxB):
    """Tính toán chỉ số IoU giữa 2 bounding box"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def refine_vehicle_class(class_name, x1, y1, x2, y2):
    """Phân loại Bus/Truck dựa trên diện tích"""
    w = x2 - x1
    h = y2 - y1
    area = w * h
    
    # Chuẩn hóa tên xe máy
    if class_name == 'motorcycle':
        return 'motorbike'

    if class_name in ['truck', 'bus']:
        # Bus thường to hơn và diện tích lớn
        if h > 0.6 * w and area > 45000:
            return 'bus'
        else:
            return 'truck'
    return class_name


def get_stable_class(track_id, track_class_votes):
    """Lấy class ổn định nhất từ lịch sử voting"""
    votes = track_class_votes.get(track_id, {})
    if not votes:
        return None
    return max(votes, key=votes.get)


# ====== PHÁT HIỆN VÀ THEO DÕI XE (LOGIC ĐẾM XE) ======
@csrf_exempt
def detect_vehicles_view(request):
    if request.method != 'POST':
        # Để test nhanh có thể bỏ qua check này hoặc giữ nguyên
        pass

    video_path = os.path.join(settings.BASE_DIR, 'main', 'static', 'img', 'road.mp4')
    if not os.path.exists(video_path):
        return JsonResponse({'error': 'Không tìm thấy video'}, status=404)

    cap = cv2.VideoCapture(video_path)
    WINDOW_NAME = "Giam sat luu luong"

    # 🔥 FIX ZOOM
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorbike': 0}
    colors = {'car': (0, 255, 0), 'truck': (255, 0, 0), 'bus': (0, 255, 255), 'motorbike': (0, 0, 255)}
    counted_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # Detect
        results = model(frame, verbose=False)[0]
        detections = []
        classes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            raw_cls = model.names[cls_id]
            
            if raw_cls in ['car', 'truck', 'bus', 'motorcycle', 'motorbike']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                final_cls = refine_vehicle_class(raw_cls, x1, y1, x2, y2)
                detections.append([x1, y1, x2, y2, conf])
                classes.append(final_cls)

        # Track
        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(dets_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            
            matched_class = None
            max_iou = 0
            for i, det in enumerate(detections):
                iou = bbox_iou(track[:4], det[:4])
                if iou > max_iou and iou > 0.3:
                    max_iou = iou
                    matched_class = classes[i]

            if matched_class:
                if track_id not in counted_ids:
                    counts[matched_class] += 1
                    counted_ids.add(track_id)

                color = colors.get(matched_class, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{matched_class} ID:{track_id}', (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return JsonResponse({'counts': counts, 'total_vehicles': sum(counts.values())})


# ====== PHÁT HIỆN TỐC ĐỘ (LOGIC MỚI + WINDOW HIỂN THỊ) ======

@csrf_exempt
def detect_vehicles_speed_view(request):
    """Xử lý video mặc định"""
    if request.method != 'POST': pass 

    video_path = os.path.join(settings.BASE_DIR, 'main', 'static', 'img', 'videotransport1.mp4')
    if not os.path.exists(video_path):
        return JsonResponse({'error': 'Không tìm thấy video'}, status=404)

    return process_video_speed(video_path)


@csrf_exempt
def detect_vehicles_speed_upload_view(request):
    """Xử lý video upload"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ'}, status=400)

    if 'video' not in request.FILES:
        return JsonResponse({'error': 'Vui lòng chọn video'}, status=400)

    uploaded_file = request.FILES['video']
    
    # Kiểm tra đuôi file
    allowed = ['.mp4', '.avi', '.mov', '.mkv']
    if not any(uploaded_file.name.lower().endswith(ext) for ext in allowed):
         return JsonResponse({'error': 'File không hỗ trợ'}, status=400)

    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
    filename = fs.save(uploaded_file.name, uploaded_file)
    video_path = fs.path(filename)

    try:
        return process_video_speed(video_path)
    finally:
        # Xóa file tạm sau khi xử lý
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass


def process_video_speed(video_path):
    """
    Hàm xử lý chính: Tracking + Speed Estimation + cv2.imshow
    Cập nhật: Màu khung xe gốc + Màu vạch gốc (Xanh/Đỏ) + Ẩn chấm vàng
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps: fps = 30.0

    WINDOW_NAME = "GIAM SAT GIAO THONG (Bam 'p' de tam dung, 'q' de thoat)"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # === MÀU GỐC CỦA BẠN ===
    colors = {
        'car': (0, 255, 0),       # Xanh lá
        'truck': (255, 0, 0),     # Xanh dương
        'bus': (0, 255, 255),     # Vàng
        'motorbike': (0, 0, 255), # Đỏ
    }
    counts = {k: 0 for k in colors}
    vehicle_speeds = []

    # Cấu hình Tracker & Model
    tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)
    MIN_TIME_PASS = 0.5 

    track_state = {}
    track_class_votes = {}
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_id += 1
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # 1. Detect
        results = model(frame, verbose=False, stream=True, conf=0.15)
        
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

        # 2. Track
        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(dets_np)

        # 3. Logic Speed
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            reference_y = y2 
            center_x = int((x1 + x2) / 2)

            # Voting Class
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

            # Speed Calculation
            track_state.setdefault(track_id, {'start_frame': None, 'done': False, 'speed': 0})
            state = track_state[track_id]

            if state['start_frame'] is None and reference_y >= LINE_START_Y and reference_y < LINE_END_Y:
                state['start_frame'] = frame_id
            
            elif state['start_frame'] is not None and not state['done'] and reference_y >= LINE_END_Y:
                time_diff = (frame_id - state['start_frame']) / fps
                
                if time_diff > MIN_TIME_PASS: 
                    speed_ms = REAL_DISTANCE / time_diff
                    speed_kmh = speed_ms * 3.6
                    
                    if 5 < speed_kmh < 90:
                        state['speed'] = speed_kmh
                        state['done'] = True
                        counts[stable_class] += 1
                        vehicle_speeds.append({
                            'id': int(track_id),
                            'type': stable_class,
                            'speed': round(speed_kmh, 1)
                        })
                    else:
                        state['start_frame'] = None 
                else:
                    state['start_frame'] = None

            # --- Drawing ---
            # Lấy màu gốc từ dictionary 'colors' đã khai báo ở trên
            color = colors.get(stable_class, (255, 255, 255))
            label = f"{stable_class} {track_id}"
            
            if state['done']:
                # Nếu đo xong thì hiện thêm tốc độ
                label += f" {state['speed']:.0f}km/h"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Đã ẩn chấm vàng (không vẽ cv2.circle nữa)

        # Vẽ vạch kẻ (Màu gốc: Blue - Red)
        cv2.line(frame, (0, LINE_START_Y), (DISPLAY_WIDTH, LINE_START_Y), (255, 0, 0), 2)
        cv2.line(frame, (0, LINE_END_Y), (DISPLAY_WIDTH, LINE_END_Y), (0, 0, 255), 2)

        # Thông tin tổng xe
        cv2.putText(frame, f"Total: {sum(counts.values())}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Hiển thị
        cv2.imshow(WINDOW_NAME, frame)
        
        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') or key == ord(' '):
            cv2.putText(frame, "PAUSED", (DISPLAY_WIDTH//2 - 100, DISPLAY_HEIGHT//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(-1)

    cap.release()
    cv2.destroyAllWindows()

    return JsonResponse({
        'counts': counts,
        'total_vehicles': sum(counts.values()),
        'vehicles': vehicle_speeds
    })