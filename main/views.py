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
    if request.method not in ['POST', 'GET']:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    video_path = os.path.join(settings.BASE_DIR, 'main', 'static', 'img', 'dovantoc1.mp4')
    if not os.path.exists(video_path):
        return JsonResponse({'error': 'Không tìm thấy video'}, status=404)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    FRAME_W, FRAME_H = 1080, 640
    roi_line1 = 300
    roi_line2 = 500
    distance_meters = 10

    counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorbike': 0}
    colors = {'car': (0, 255, 0), 'truck': (255, 0, 0), 'bus': (0, 255, 255), 'motorbike': (0, 0, 255)}

    # Khởi tạo lại tracker bên trong hàm để reset mỗi lần chạy
    local_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.25)

    track_times = {}
    track_class_history = {} 
    track_final_class = {}   
    counted_ids = set()
    vehicle_speeds = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame_id += 1

        results = model(frame)[0]
        detections, classes = [], []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in counts:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])
                classes.append(cls_name)

        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = local_tracker.update(dets_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            center_y = int((y1 + y2) / 2)

            # 1. Tìm class từ YOLO
            matched_class = None
            max_iou = 0
            for i, det in enumerate(detections):
                iou = bbox_iou(track[:4], det[:4])
                if iou > max_iou:
                    max_iou = iou
                    matched_class = classes[i]

            if matched_class is None: continue

            # 2. Cập nhật lịch sử và lấy class phổ biến nhất (Voting)
            history = track_class_history.setdefault(track_id, [])
            history.append(matched_class)
            if len(history) > 20: history.pop(0) # Giữ 20 frame gần nhất
            
            voted_class = max(set(history), key=history.count)

            # 3. Logic KHÓA CLASS: Chống nhảy từ Bus/Truck sang Car
            if track_id not in track_final_class:
                track_final_class[track_id] = voted_class
            else:
                # Nếu đã từng được xác định là xe lớn, không cho phép đổi lại thành xe nhỏ
                current_status = track_final_class[track_id]
                if current_status not in ['truck', 'bus'] and voted_class in ['truck', 'bus']:
                    track_final_class[track_id] = voted_class
            
            # Luôn sử dụng nhãn đã được "khóa" để hiển thị
            final_type = track_final_class[track_id]

            # 4. Tính vận tốc
            track_times.setdefault(track_id, {})
            prev_y = track_times[track_id].get('prev_y', center_y)

            if prev_y < roi_line1 <= center_y and 't1' not in track_times[track_id]:
                track_times[track_id]['t1'] = frame_id

            if prev_y < roi_line2 <= center_y and 't1' in track_times[track_id] and 't2' not in track_times[track_id]:
                track_times[track_id]['t2'] = frame_id

                if track_id not in counted_ids:
                    duration_sec = (track_times[track_id]['t2'] - track_times[track_id]['t1']) / fps
                    if 0.3 <= duration_sec <= 10:
                        speed_kmh = (distance_meters / duration_sec) * 3.6
                        
                        vehicle_speeds.append({
                            'id': track_id, 
                            'type': final_type, 
                            'speed': round(speed_kmh, 1)
                        })
                        counts[final_type] += 1
                        counted_ids.add(track_id)

                        Vehicle.objects.update_or_create(
                            track_id=track_id,
                            defaults={'vehicle_type': final_type, 'speed': round(speed_kmh, 1)}
                        )

            track_times[track_id]['prev_y'] = center_y

            # 5. Vẽ đồ họa
            color = colors.get(final_type, (255, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{final_type} #{track_id}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Vẽ vạch đo
        cv2.line(frame, (0, roi_line1), (FRAME_W, roi_line1), (255, 0, 0), 2)
        cv2.line(frame, (0, roi_line2), (FRAME_W, roi_line2), (0, 0, 255), 2)
        
        cv2.imshow("Giam sat toc do", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    return JsonResponse({'counts': counts, 'total_vehicles': sum(counts.values()), 'vehicles': vehicle_speeds})