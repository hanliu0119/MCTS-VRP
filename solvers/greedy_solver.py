import numpy as np
from collections import defaultdict
from copy import deepcopy

class Vehicle:
    def __init__(self, vid, capacity=12):
        self.id = vid
        self.capacity = capacity
        self.load = 0
        self.time = 0
        self.location = None  # set when first request is assigned
        self.route = []  # list of (pickup_id, dropoff_id)

    # Checks if the vehicle can serve a request within the pickup time window
    def can_serve(self, request, travel_time_matrix):
        pickup_id = request.pickup_id
        dropoff_id = request.dropoff_id

        if self.location is None:
            travel_time = 0
        else:
            travel_time = travel_time_matrix[self.location][pickup_id]

        arrival_time = max(self.time + travel_time, request.pickup_tw[0])
        if arrival_time > request.pickup_tw[1]:
            return False  # too late

        return True

    # Assigns a request to the vehicle and updates its state and route
    def assign(self, request, travel_time_matrix):
        pickup_id = request.pickup_id
        dropoff_id = request.dropoff_id

        if self.location is None:
            travel_to_pickup = 0
        else:
            travel_to_pickup = travel_time_matrix[self.location][pickup_id]

        # update to pickup
        self.time = max(self.time + travel_to_pickup, request.pickup_tw[0]) + request.service_time
        self.location = pickup_id
        self.load += request.demand

        # add pickup to route
        self.route.append(pickup_id)

        # travel to dropoff
        travel_to_dropoff = travel_time_matrix[pickup_id][dropoff_id]
        self.time = max(self.time + travel_to_dropoff, request.dropoff_tw[0]) + request.service_time
        self.location = dropoff_id
        self.load -= request.demand

        # add dropoff to route
        self.route.append(dropoff_id)


# Solves the routing problem greedily by assigning earliest-feasible requests to available vehicles
def greedy_solve(requests, travel_time_matrices, vehicle_count=4, capacity=12):
    # Group requests by date
    requests_by_date = defaultdict(list)
    for req in requests:
        requests_by_date[req.date].append(req)

    vehicles_by_date = {}
    unserved_by_date = {}

    for date, reqs in requests_by_date.items():
        travel_time_matrix = travel_time_matrices[date]
        vehicles = [Vehicle(vid=i, capacity=capacity) for i in range(vehicle_count)]
        unserved = []

        # Sort requests by pickup time
        reqs = sorted(reqs, key=lambda r: r.pickup_tw[0])

        for req in reqs:
            assigned = False
            for v in vehicles:
                if v.can_serve(req, travel_time_matrix):
                    v.assign(req, travel_time_matrix)
                    assigned = True
                    break
            if not assigned:
                unserved.append(req)

        vehicles_by_date[date] = vehicles
        unserved_by_date[date] = unserved

    return vehicles_by_date, unserved_by_date
