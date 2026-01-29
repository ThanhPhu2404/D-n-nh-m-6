import os
import cv2
import numpy as np
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

# ===== DISPLAY CONFIG (FIX ZOOM OPENCV) =====
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# ====== VIEW TRANG ======
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


# ====== XỬ LÝ TÌM ĐƯỜNG NGẮN NHẤT ======
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


# ====== PHÁT HIỆN VÀ THEO DÕI XE ======

# Khởi tạo model YOLO
model_path = os.path.join(settings.BASE_DIR, 'yolov5su.pt')
model = YOLO(model_path)

# Khởi tạo tracker SORT
tracker = Sort()


def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


@csrf_exempt
def detect_vehicles_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ'}, status=400)

    video_path = os.path.join(settings.BASE_DIR, 'main', 'static', 'img', 'road.mp4')
    if not os.path.exists(video_path):
        return JsonResponse({'error': 'Không tìm thấy video'}, status=404)

    cap = cv2.VideoCapture(video_path)

    WINDOW_NAME = "Giam sat giao thong"

    # 🔥 FIX ZOOM – BẮT BUỘC
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorbike': 0}
    colors = {
        'car': (0, 255, 0),
        'truck': (255, 0, 0),
        'bus': (0, 255, 255),
        'motorbike': (0, 0, 255),
    }

    counted_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ RESIZE FRAME
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        results = model(frame)[0]
        detections, classes = [], []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name in counts:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])
                classes.append(class_name)

        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(dets_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)

            matched_class, max_iou = None, 0
            for i, det in enumerate(detections):
                iou = bbox_iou(track[:4], det[:4])
                if iou > max_iou:
                    max_iou = iou
                    matched_class = classes[i]

            if matched_class and track_id not in counted_ids:
                counts[matched_class] += 1
                counted_ids.add(track_id)

            color = colors.get(matched_class, (0, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f'{matched_class} ID:{track_id}',
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # ❌ KHÔNG DÙNG STRING TRỰC TIẾP
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return JsonResponse({
        'counts': counts,
        'total_vehicles': sum(counts.values())
    })


@csrf_exempt
def detect_vehicles_speed_view(request):
    """
    Phân tích video mặc định (videotransport1.mp4)
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ'}, status=400)

    video_path = os.path.join(
        settings.BASE_DIR, 'main', 'static', 'img', 'videotransport1.mp4'
    )

    if not os.path.exists(video_path):
        return JsonResponse({'error': 'Không tìm thấy video'}, status=404)

    return process_video_speed(video_path)


@csrf_exempt
def detect_vehicles_speed_upload_view(request):
    """
    Phân tích video do người dùng upload
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ'}, status=400)

    # Kiểm tra có file upload không
    if 'video' not in request.FILES:
        return JsonResponse({'error': 'Vui lòng chọn video để upload'}, status=400)

    uploaded_file = request.FILES['video']
    
    # Kiểm tra định dạng file
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_ext not in allowed_extensions:
        return JsonResponse({
            'error': f'Định dạng file không được hỗ trợ. Chỉ chấp nhận: {", ".join(allowed_extensions)}'
        }, status=400)

    # Lưu file vào thư mục uploads
    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
    filename = fs.save(uploaded_file.name, uploaded_file)
    video_path = fs.path(filename)

    try:
        # Xử lý video
        result = process_video_speed(video_path)
        return result
    finally:
        # Xóa file sau khi xử lý xong (tùy chọn)
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass


def process_video_speed(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    WINDOW_NAME = "Giam sat toc do"

    # 🔥 FIX ZOOM – CỬA SỔ
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    roi_line1 = 350
    roi_line2 = 550
    distance_meters = 20

    colors = {
        'car': (0, 255, 0),
        'truck': (255, 0, 0),
        'bus': (0, 255, 255),
        'motorbike': (0, 0, 255),
    }
    counts = {k: 0 for k in colors}

    tracker = Sort(max_age=30, min_hits=3)
    track_state = {}
    track_class_votes = {}
    vehicle_speeds = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # ✅ RESIZE FRAME
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        results = model(frame)[0]
        detections, classes = [], []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            raw_class = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            class_name = refine_vehicle_class(raw_class, x1, y1, x2, y2)
            if class_name in counts:
                detections.append([x1, y1, x2, y2, conf])
                classes.append(class_name)

        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(dets_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            center_y = (y1 + y2) // 2

            matched_class, max_iou = None, 0
            for i, det in enumerate(detections):
                iou = bbox_iou(track[:4], det[:4])
                if iou > max_iou and iou > 0.3:
                    max_iou = iou
                    matched_class = classes[i]

            if matched_class:
                track_class_votes.setdefault(track_id, {})
                track_class_votes[track_id][matched_class] = \
                    track_class_votes[track_id].get(matched_class, 0) + 1

            stable_class = get_stable_class(track_id, track_class_votes)
            if stable_class not in counts:
                continue

            track_state.setdefault(track_id, {'t1': None, 'done': False})
            state = track_state[track_id]

            if state['t1'] is None and center_y >= roi_line1:
                state['t1'] = frame_id

            if state['t1'] and not state['done'] and center_y >= roi_line2:
                duration = (frame_id - state['t1']) / fps
                speed = (distance_meters / duration) * 3.6 if duration > 0 else 0

                vehicle_speeds.append({
                    'id': track_id,
                    'type': stable_class,
                    'speed': round(speed, 1)
                })

                counts[stable_class] += 1
                state['done'] = True

            color = colors[stable_class]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f'{stable_class} ID:{track_id}',
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.line(frame, (0, roi_line1), (DISPLAY_WIDTH, roi_line1), (255, 0, 0), 2)
        cv2.line(frame, (0, roi_line2), (DISPLAY_WIDTH, roi_line2), (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return JsonResponse({
        'counts': counts,
        'total_vehicles': sum(counts.values()),
        'vehicles': vehicle_speeds
    })


def refine_vehicle_class(class_name, x1, y1, x2, y2):
    """
    Phân biệt lại truck / bus dựa trên kích thước bbox
    """
    w = x2 - x1
    h = y2 - y1
    area = w * h

    if class_name in ['truck', 'bus']:
        # Bus: cao và to
        if h > 0.6 * w and area > 60000:
            return 'bus'
        else:
            return 'truck'

    return class_name


def get_stable_class(track_id, track_class_votes):
    """
    Lấy class ổn định nhất theo voting
    """
    votes = track_class_votes.get(track_id, {})
    if not votes:
        return None
    return max(votes, key=votes.get)