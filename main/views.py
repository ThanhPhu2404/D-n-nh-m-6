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

    counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorbike': 0}
    colors = {
        'car': (0, 255, 0),
        'truck': (255, 0, 0),
        'bus': (0, 255, 255),
        'motorbike': (0, 0, 255),
    }

    counted_ids = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame)[0]

        detections = []
        classes = []

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
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)

            matched_class = None
            max_iou = 0
            for i, det in enumerate(detections):
                iou = bbox_iou(track[:4], det[:4])
                if iou > max_iou:
                    max_iou = iou
                    matched_class = classes[i]

            if matched_class and track_id not in counted_ids and matched_class in counts:
                counts[matched_class] += 1
                counted_ids.add(track_id)

                Vehicle.objects.get_or_create(
                    track_id=track_id,
                    defaults={
                        'license_plate': None,
                        'vehicle_type': matched_class,
                        'speed': None
                    }
                )

            # Vẽ khung và nhãn
            color = colors.get(matched_class, (0, 0, 255))
            label = f'{matched_class} ID:{track_id}'

            # ✅ Điều chỉnh kích thước khung bao cho phù hợp
            box_w = x2 - x1
            box_h = y2 - y1
            shrink_factor = 0.1  # Co lại 10%
            x1 += int(box_w * shrink_factor)
            x2 -= int(box_w * shrink_factor)
            y1 += int(box_h * shrink_factor)
            y2 -= int(box_h * shrink_factor)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Hiển thị khung hình
        cv2.imshow("Giám sát giao thông", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    response_data = {
        'counts': counts,
        'total_vehicles': sum(counts.values())
    }

    return JsonResponse(response_data)


@csrf_exempt
def detect_vehicles_speed_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ'}, status=400)

    video_path = os.path.join(
        settings.BASE_DIR, 'main', 'static', 'img', 'videotransport.mp4'
    )

    if not os.path.exists(video_path):
        return JsonResponse({'error': 'Không tìm thấy video'}, status=404)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # ===== CẤU HÌNH =====
    roi_line1 = 350
    roi_line2 = 550
    distance_meters = 20

    colors = {
        'car': (0, 255, 0),
        'truck': (255, 0, 0),
        'bus': (0, 255, 255),
        'motorbike': (0, 0, 255),
    }
    counts = {k: 0 for k in colors.keys()}

    tracker = Sort(max_age=30, min_hits=3)

    track_state = {}          # {id: {'t1': frame, 'done': bool}}
    track_class_votes = {}    # {id: {'car':3, 'bus':7}}
    vehicle_speeds = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        results = model(frame)[0]

        detections = []
        classes = []

        # ===== DETECTION + REFINE =====
        for box in results.boxes:
            cls_id = int(box.cls[0])
            raw_class = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            class_name = refine_vehicle_class(
                raw_class, x1, y1, x2, y2
            )

            if class_name in counts:
                detections.append([x1, y1, x2, y2, conf])
                classes.append(class_name)

        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(dets_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            center_y = int((y1 + y2) / 2)

            # ===== MATCH CLASS + VOTING =====
            matched_class = None
            max_iou = 0
            for i, det in enumerate(detections):
                iou = bbox_iou(track[:4], det[:4])
                if iou > max_iou and iou > 0.3:
                    max_iou = iou
                    matched_class = classes[i]

            if matched_class:
                if track_id not in track_class_votes:
                    track_class_votes[track_id] = {}
                track_class_votes[track_id][matched_class] = \
                    track_class_votes[track_id].get(matched_class, 0) + 1

            stable_class = get_stable_class(track_id, track_class_votes)
            if stable_class not in counts:
                continue

            # ===== INIT STATE =====
            if track_id not in track_state:
                track_state[track_id] = {
                    't1': None,
                    'done': False
                }

            state = track_state[track_id]

            # ===== QUA LINE 1 =====
            if state['t1'] is None and center_y >= roi_line1:
                state['t1'] = frame_id

            # ===== QUA LINE 2 → ĐẾM + TỐC ĐỘ =====
            if (
                state['t1'] is not None
                and not state['done']
                and center_y >= roi_line2
            ):
                t1 = state['t1']
                t2 = frame_id
                duration = (t2 - t1) / fps

                speed_kmh = (distance_meters / duration) * 3.6 if duration > 0 else 0

                vehicle_speeds.append({
                    'id': track_id,
                    'type': stable_class,
                    'speed': round(speed_kmh, 1)
                })

                counts[stable_class] += 1
                state['done'] = True

            # ===== VẼ (DÙNG CLASS ỔN ĐỊNH) =====
            color = colors.get(stable_class, (0, 0, 255))
            label = f'{stable_class} ID:{track_id}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                frame, label, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # ===== VẼ ROI =====
        cv2.line(frame, (0, roi_line1), (frame.shape[1], roi_line1), (255, 0, 0), 2)
        cv2.line(frame, (0, roi_line2), (frame.shape[1], roi_line2), (0, 0, 255), 2)

        cv2.imshow("Giam sat toc do", frame)
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
