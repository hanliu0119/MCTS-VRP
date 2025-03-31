import math
import random
import copy
from models.mdp import MDPState
from models.generative_model import sample_future_requests


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action
        self.visits = 0
        self.total_value = 0.0

    def is_fully_expanded(self):
        legal_actions = self.state.get_legal_actions()
        return len(self.children) == len(legal_actions)

    def best_child(self, c_param=0.7):
        if not self.children:
            return None
        return max(self.children, key=lambda node: node.uct_value(c_param))

    def uct_value(self, c):
        if self.visits == 0:
            return float('inf')
        return (self.total_value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def expand(self, top_k=30):
        tried_actions = [child.action for child in self.children]
        legal_actions = self.state.get_legal_actions()

        # Score and rank actions
        scored_actions = []
        for (vi, ri) in legal_actions:
            vehicle = self.state.vehicles[vi]
            request = self.state.pending_requests[ri]

            flex = request.pickup_tw[1] - request.pickup_tw[0]
            urgency = max(0, 7200 - request.pickup_tw[0])
            travel_to = 0 if vehicle.location is None else self.state.travel_time_matrix[vehicle.location][request.pickup_id]
            proximity_bonus = -travel_to * 0.1
            load_penalty = len(vehicle.route) * 0.1
            load_ratio = vehicle.load / vehicle.capacity

            score = urgency + flex * 0.01 + proximity_bonus - load_penalty - load_ratio * 2
            scored_actions.append(((vi, ri), score))

        sorted_actions = sorted(scored_actions, key=lambda x: x[1], reverse=True)

        for (action, _) in sorted_actions[:top_k]:
            if action not in tried_actions:
                new_state = self.state.apply_action(action)
                child_node = MCTSNode(state=new_state, parent=self, action=action)
                self.children.append(child_node)
                return child_node

        return None

    def is_terminal(self):
        return self.state.is_terminal()


class MCTS:
    def __init__(self, root_state, model=None, rollout_depth=20, rollout_count=5):
        self.root = MCTSNode(state=root_state)
        self.model = model
        self.rollout_depth = rollout_depth
        self.rollout_count = rollout_count

    def search(self, iterations):
        rewards = []

        for i in range(iterations):
            node = self.select(self.root)
            if not node.is_terminal():
                node = node.expand()
                if node is None:
                    continue
            value = self.simulate(node.state)
            rewards.append(value)
            self.backpropagate(node, value)

            if i % 100 == 0:
                recent = rewards[-100:]
                print(f"Rollout {i}: Reward = {value:.4f}, Avg (last 100) = {sum(recent)/len(recent):.4f}, Max = {max(recent):.4f}")

        return self.root.best_child(c_param=0.0).action if self.root.children else None

    def select(self, node):
        while not node.is_terminal() and node.is_fully_expanded():
            next_node = node.best_child()
            if next_node is None:
                break
            node = next_node
        return node

    def simulate(self, state):
        sim_state = state.copy()

        for step in range(self.rollout_depth):
            legal = sim_state.get_legal_actions()
            if not legal:
                future_requests = sample_future_requests(self.model, sim_state.pending_requests, n=10)
                sim_state.pending_requests.extend(future_requests)
                legal = sim_state.get_legal_actions()
                if not legal:
                    break

            scored_actions = []
            for (vi, ri) in legal:
                vehicle = sim_state.vehicles[vi]
                request = sim_state.pending_requests[ri]

                flex = request.pickup_tw[1] - request.pickup_tw[0]
                urgency = max(0, 7200 - request.pickup_tw[0])
                travel_to = 0 if vehicle.location is None else sim_state.travel_time_matrix[vehicle.location][request.pickup_id]
                proximity_bonus = -travel_to * 0.1
                load_penalty = len(vehicle.route) * 0.1
                load_ratio = vehicle.load / vehicle.capacity

                # Add step penalty to encourage early action
                score = urgency + flex * 0.01 + proximity_bonus - load_penalty - load_ratio * 2 - step * 0.1
                scored_actions.append(((vi, ri), score))

            if not scored_actions:
                break

            best_action = max(scored_actions, key=lambda x: x[1])[0]
            sim_state = sim_state.apply_action(best_action)

        idle_penalty = sum(v.idle_time for v in sim_state.vehicles) * 0.0001
        travel_penalty = sum(v.get_penalty(sim_state.travel_time_matrix) for v in sim_state.vehicles)
        return sim_state.reward() - idle_penalty - travel_penalty

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_value += reward
            node = node.parent
