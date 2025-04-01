import math
import random
import copy
import numpy as np

from models.mdp import MDPState
from models.generative_model import ClusteredRequestGenerator


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action
        self.visits = 0
        self.total_value = 0.0

    # Returns True if all legal actions have been expanded
    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    # Returns the child with the highest UCT value
    def best_child(self, c_param=1.0):
        if not self.children:
            return None
        return max(self.children, key=lambda node: node.uct_value(c_param))

    # Computes the UCT value for this node
    def uct_value(self, c):
        if self.visits == 0:
            return float('inf')
        return (self.total_value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    # Expands the node by adding one of the top-k highest scoring child nodes
    def expand(self, top_k=30):
        tried_actions = [child.action for child in self.children]
        legal_actions = self.state.get_legal_actions()

        scored = []
        for (vi, ri) in legal_actions:
            vehicle = self.state.vehicles[vi]
            request = self.state.pending_requests[ri]
            flex = request.pickup_tw[1] - request.pickup_tw[0]
            urgency = max(0, 7200 - request.pickup_tw[0])
            travel_to = 0 if vehicle.location is None else self.state.travel_time_matrix[vehicle.location][request.pickup_id]
            score = urgency + flex * 0.01 - travel_to * 0.1
            scored.append(((vi, ri), score))

        for (action, _) in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]:
            if action not in tried_actions:
                new_state = self.state.apply_action(action)
                node = MCTSNode(new_state, self, action)
                self.children.append(node)
                return node
        return None

    # Checks if this node represents a terminal state
    def is_terminal(self):
        return self.state.is_terminal()


class MCTS:
    def __init__(self, root_state, model: ClusteredRequestGenerator, rollout_depth=20):
        self.root = MCTSNode(state=root_state)
        self.model = model
        self.rollout_depth = rollout_depth

    # Performs MCTS for a given number of iterations and returns the best action
    def search(self, iterations):
        rewards = []
        for i in range(iterations):
            node = self.select(self.root)
            if node is None:
                continue
            if not node.is_terminal():
                expanded = node.expand()
                if expanded:
                    node = expanded
            value = self.simulate(node.state)
            self.backpropagate(node, value)
            rewards.append(value)

        best = self.root.best_child(c_param=0.0)
        return best.action if best else None

    # Selects a node to expand by traversing fully expanded nodes
    def select(self, node):
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
            if node is None:
                break
        return node

    # Simulates a rollout from the given state and returns a shaped reward
    def simulate(self, state):
        sim_state = state.copy()
        self._inject_synthetic_requests(sim_state, n=120)

        served = 0
        for step in range(self.rollout_depth):
            legal = sim_state.get_legal_actions()
            if not legal:
                break

            scores = []
            for (vi, ri) in legal:
                vehicle = sim_state.vehicles[vi]
                request = sim_state.pending_requests[ri]
                flex = request.pickup_tw[1] - request.pickup_tw[0]
                urgency = max(0, 7200 - request.pickup_tw[0])
                travel = 0 if vehicle.location is None else sim_state.travel_time_matrix[vehicle.location][request.pickup_id]
                score = urgency + flex * 0.01 - travel * 0.1 - step * 0.1
                scores.append(((vi, ri), score))

            if not scores:
                break

            action = max(scores, key=lambda x: x[1])[0]
            sim_state = sim_state.apply_action(action)
            served += 1

        reward = served
        travel_penalty = sum(v.get_penalty(sim_state.travel_time_matrix) for v in sim_state.vehicles)
        idle_penalty = sum(v.idle_time for v in sim_state.vehicles) * 0.0001
        overload_penalty = sum(max(0, v.load - v.capacity) for v in sim_state.vehicles) * 2.0
        balance_penalty = np.std([v.load for v in sim_state.vehicles]) * 0.2

        return reward - travel_penalty - idle_penalty - overload_penalty - balance_penalty

    # Adds synthetic future requests using the generative model
    def _inject_synthetic_requests(self, sim_state, n=120):
        new_requests = []
        for _ in range(n):
            if not sim_state.pending_requests:
                break
            base = random.choice(sim_state.pending_requests)
            plat, plon = base.pickup_loc
            pt = base.pickup_tw[0] + random.randint(-300, 300)
            cluster = self.model.predict_cluster(plat, plon, pt)
            dlat, dlon = self.model.get_dropoff_center(cluster)
            delay = 900 + random.randint(-300, 300)

            new_requests.append(base.__class__(
                date="future",
                pickup_id=0,
                pickup_loc=(plat, plon),
                pickup_tw=(pt, pt + 1800),
                dropoff_id=0,
                dropoff_loc=(dlat, dlon),
                dropoff_tw=(pt + delay, pt + delay + 1800),
                demand=base.demand,
                service_time=60
            ))
        sim_state.pending_requests.extend(new_requests)

    # Backpropagates the reward through the path to the root
    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_value += reward
            node = node.parent
