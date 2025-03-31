import os
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class Request:
    date: str
    pickup_id: int
    pickup_loc: tuple
    pickup_tw: tuple
    dropoff_id: int
    dropoff_loc: tuple
    dropoff_tw: tuple
    demand: int
    service_time: int

def load_requests(file_path, date):
    requests = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    tasks = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        task = {
            'id': int(parts[0]),
            'x': float(parts[1]),
            'y': float(parts[2]),
            'demand': int(parts[3]),
            'early': int(parts[4]),
            'late': int(parts[5]),
            'service': int(parts[6]),
            'pickup_idx': int(parts[7]),
            'delivery_idx': int(parts[8])
        }
        tasks.append(task)

    visited = set()
    for task in tasks:
        if task['demand'] > 0 and task['id'] not in visited:
            pickup = task
            delivery = next(t for t in tasks if t['id'] == pickup['delivery_idx'])
            visited.add(pickup['id'])
            visited.add(delivery['id'])

            req = Request(
                date=date,
                pickup_id=pickup['id'],
                pickup_loc=(pickup['x'], pickup['y']),
                pickup_tw=(pickup['early'], pickup['late']),
                dropoff_id=delivery['id'],
                dropoff_loc=(delivery['x'], delivery['y']),
                dropoff_tw=(delivery['early'], delivery['late']),
                demand=pickup['demand'],
                service_time=pickup['service']
            )
            requests.append(req)

    return requests

def load_all_requests(data_root):
    all_requests = []
    distance_matrices = {}
    travel_time_matrices = {}
    request_count_by_date = defaultdict(int)

    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue

        hexaly_file = None
        for file in os.listdir(folder_path):
            if file.startswith("hexaly") and file.endswith(".txt"):
                hexaly_file = file
                break

        if not hexaly_file:
            continue

        # Load requests
        file_path = os.path.join(folder_path, hexaly_file)
        requests = load_requests(file_path, date=folder)
        request_count_by_date[folder] += len(requests)
        all_requests.extend(requests)

        # Load matrices
        try:
            d_matrix = np.load(os.path.join(folder_path, f"distance_matrix_{folder}.npy"))
            t_matrix = np.load(os.path.join(folder_path, f"travel_time_matrix_{folder}.npy"))
            distance_matrices[folder] = d_matrix
            travel_time_matrices[folder] = t_matrix
        except FileNotFoundError as e:
            print(f"⚠️ Warning: Missing matrix for {folder} — {e}")

    for date, count in sorted(request_count_by_date.items()):
        print(f"{date}: loaded {count} requests")

    return all_requests, distance_matrices, travel_time_matrices
