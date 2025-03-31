import copy

class VehicleState:
    def __init__(self, vid, capacity=12, location=None, time=0):
        self.id = vid
        self.capacity = capacity
        self.load = 0
        self.location = location
        self.time = time
        self.route = []
        self.idle_time = 0
        self.assignments = 0  # For load balancing

    def clone(self):
        return copy.deepcopy(self)

    def can_serve(self, request, travel_time_matrix):
        N = 10
        pickup_id = request.pickup_id
        dropoff_id = request.dropoff_id
        allow_late = 600 * N

        travel_to_pickup = 0 if self.location is None else travel_time_matrix[self.location][pickup_id]
        arrival_at_pickup = max(self.time + travel_to_pickup, request.pickup_tw[0])
        if arrival_at_pickup > request.pickup_tw[1]:
            return False

        travel_to_dropoff = travel_time_matrix[pickup_id][dropoff_id]
        arrival_at_dropoff = arrival_at_pickup + request.service_time + travel_to_dropoff
        if arrival_at_dropoff > request.dropoff_tw[1] + allow_late:
            return False

        return self.load + request.demand <= self.capacity

    def apply(self, request, travel_time_matrix):
        pickup_id = request.pickup_id
        dropoff_id = request.dropoff_id

        travel_to_pickup = 0 if self.location is None else travel_time_matrix[self.location][pickup_id]
        self.idle_time += max(0, request.pickup_tw[0] - self.time - travel_to_pickup)

        self.time = max(self.time + travel_to_pickup, request.pickup_tw[0]) + request.service_time
        self.location = pickup_id
        self.load += request.demand
        self.route.append(pickup_id)

        travel_to_dropoff = travel_time_matrix[pickup_id][dropoff_id]
        self.time = max(self.time + travel_to_dropoff, request.dropoff_tw[0]) + request.service_time
        self.location = dropoff_id
        self.load -= request.demand
        self.route.append(dropoff_id)

        self.assignments += 1  # For load balancing

    def get_penalty(self, travel_time_matrix):
        penalty_idle = self.idle_time * 0.0001
        penalty_travel = 0
        for i in range(len(self.route) - 1):
            from_id = self.route[i]
            to_id = self.route[i + 1]
            penalty_travel += travel_time_matrix[from_id][to_id] * 0.00001
        return penalty_idle + penalty_travel


class MDPState:
    def __init__(self, vehicles, pending_requests, travel_time_matrix):
        self.vehicles = [v.clone() for v in vehicles]
        self.pending_requests = pending_requests.copy()
        self.travel_time_matrix = travel_time_matrix
        self.total_reward = 0

    def is_terminal(self):
        return len(self.pending_requests) == 0

    def get_legal_actions(self):
        actions = []
        for i, vehicle in enumerate(self.vehicles):
            for j, req in enumerate(self.pending_requests):
                if vehicle.can_serve(req, self.travel_time_matrix):
                    actions.append((i, j))
        return actions

    def apply_action(self, action):
        vi, ri = action
        next_state = self.copy()
        vehicle = next_state.vehicles[vi]
        request = next_state.pending_requests.pop(ri)
        vehicle.apply(request, next_state.travel_time_matrix)
        next_state.total_reward = self.total_reward + 1
        return next_state

    def copy(self):
        return MDPState(self.vehicles, self.pending_requests, self.travel_time_matrix)

    def reward(self):
        return self.total_reward
