from models.data_loader import load_all_requests
from models.generative_model import train_model, sample_future_requests
from evaluation.request_distribution import (
    plot_time_window_gantt,
    plot_hourly_demand,
    plot_request_location
)
from solvers.greedy_solver import greedy_solve
from evaluation.greedy_evaluation import plot_vehicle_request_map
from models.mdp import VehicleState, MDPState
from solvers.mcts_solver import MCTS

from collections import defaultdict

def main():
    # # load data
    requests, distance_matrices, travel_time_matrices = load_all_requests("data")
    # for req in requests[:3]:
    #     print(req)

    # Run all real request distribution plots
    plot_time_window_gantt(requests, "real")
    plot_hourly_demand(requests, "real")
    plot_request_location(requests, "real")

    # train NN model to generate future request
    model = train_model(requests)
    future_requests = sample_future_requests(model, requests, n=120)
    print(future_requests[0])
    print(f"Sampled {len(future_requests)} synthetic requests for 'future_date'")
    plot_time_window_gantt(future_requests, "sampled")
    plot_hourly_demand(future_requests, "sampled")
    plot_request_location(future_requests, "sampled")

    # test greedy
    vehicles_by_date, unserved_by_date = greedy_solve(requests, travel_time_matrices)

    for date in vehicles_by_date:
        total_assigned = sum(len(v.route) // 2 for v in vehicles_by_date[date])
        print(f"{date}: Assigned {total_assigned} requests, Unserved: {len(unserved_by_date[date])}")
    # evaluate greedy
    requests_by_date = defaultdict(list)
    for r in requests:
        requests_by_date[r.date].append(r)

    plot_vehicle_request_map(vehicles_by_date, requests_by_date)

    # test MCTS
    mcts_vehicles_by_date = {}
    mcts_unserved_by_date = {}

    # Group requests by date
    requests_by_date = defaultdict(list)
    for r in requests:
        requests_by_date[r.date].append(r)

    #  Train a single global generative model using all requests
    print("Training global generative model...")
    model = train_model(requests)

    # Loop over all dates
    for date in sorted(requests_by_date):
        print(f"\n Running MCTS on {date}")
        day_requests = requests_by_date[date]
        travel_time_matrix = travel_time_matrices[date]

        # Init empty vehicles
        vehicles = [VehicleState(i, capacity=12) for i in range(4)]
        state = MDPState(vehicles, day_requests, travel_time_matrix)

        step = 0
        while not state.is_terminal():
            print(f"\n MCTS Step {step} — Pending Requests: {len(state.pending_requests)}")

            # Log number of legal actions for debugging
            legal_actions = state.get_legal_actions()
            print(f" Legal actions available: {len(legal_actions)}")

            if len(legal_actions) == 0:
                print(" No legal actions available — breaking early.")
                break

            mcts = MCTS(state, model=model, rollout_depth=20)
            action = mcts.search(iterations=500)
            if action is None:
                break
            print(f" Assigning request {action[1]} to vehicle {action[0]}")
            state = state.apply_action(action)
            step += 1

        print(f"{date}: Served = {state.reward()}, Unserved = {len(state.pending_requests)}")

        mcts_vehicles_by_date[date] = state.vehicles
        mcts_unserved_by_date[date] = state.pending_requests
        break  # ← remove this to run on all dates

if __name__ == "__main__":
    main()
