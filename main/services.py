import heapq

def dijkstra(nodes, edges, start_id, end_id):
    """
    nodes: danh sách Node objects
    edges: danh sách Edge objects
    start_id, end_id: id của điểm bắt đầu và điểm kết thúc

    Trả về list node id theo đường đi ngắn nhất, hoặc [] nếu không tìm thấy.
    """

    # Tạo đồ thị dưới dạng adjacency list: node_id -> [(neighbor_id, weight), ...]
    graph = {node.id: [] for node in nodes}
    for edge in edges:
        graph[edge.start_id].append((edge.end_id, edge.weight))
        # Nếu đường hai chiều, thêm dòng này:
        graph[edge.end_id].append((edge.start_id, edge.weight))

    # Khoảng cách từ start đến mỗi node
    distances = {node.id: float('inf') for node in nodes}
    distances[start_id] = 0

    # Lưu đường đi: node_id -> node_id trước đó trên đường đi ngắn nhất
    previous_nodes = {}

    # Min-heap priority queue: (distance, node_id)
    queue = [(0, start_id)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == end_id:
            break  # Đã tìm thấy đường đi ngắn nhất đến đích

        if current_distance > distances[current_node]:
            continue  # Đã có đường đi tốt hơn

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # Lấy đường đi ngược từ end đến start
    path = []
    current = end_id
    while current != start_id:
        path.append(current)
        current = previous_nodes.get(current)
        if current is None:
            return []  # Không có đường đi
    path.append(start_id)
    path.reverse()
    return path
