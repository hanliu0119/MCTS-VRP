from models.data_loader import load_all_requests
from models.generative_model import ClusteredRequestGenerator
from evaluation.request_distribution import (
    plot_time_window_gantt,
    plot_hourly_demand,
    plot_request_location
)
from models.data_loader import Request
from solvers.greedy_solver import greedy_solve
from models.mdp import VehicleState, MDPState
from solvers.mcts_solver import MCTS

from collections import defaultdict
import random 
from collections import defaultdict
from models.generative_model import ClusteredRequestGenerator
from models.data_loader import Request  # Adjust import if needed
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from evaluation.generaitve_model_evaluation import evaluate_model_fit
# from evaluation.greedy_evaluation import plot_vehicle_request_map
# from models.generative_model import train_model, sample_future_requests

def generate_synthetic_requests_clustered(requests, n_samples=120, n_clusters=30, fixed_delay=1200):
    gen_model = ClusteredRequestGenerator(n_clusters=n_clusters)
    gen_model.fit(requests)

    future_requests = []
    for _ in range(n_samples):
        base = random.choice(requests)
        plat, plon = base.pickup_loc
        pt = base.pickup_tw[0]

        cluster = gen_model.predict_cluster(plat, plon, pt)
        dlat, dlon = gen_model.get_dropoff_center(cluster)

        dropoff_time = pt + fixed_delay
        future_requests.append(Request(
            date="future_date",
            pickup_id=0,
            pickup_loc=(plat, plon),
            pickup_tw=(pt, pt + 1800),
            dropoff_id=0,
            dropoff_loc=(dlat, dlon),
            dropoff_tw=(dropoff_time, dropoff_time + 1800),
            demand=base.demand,
            service_time=60
        ))

    print(future_requests[0])
    print(f"Sampled {len(future_requests)} synthetic requests for 'future_date'")
    return future_requests

def run_mcts_with_clustered_model(
    requests,
    travel_time_matrices,
    vehicle_capacity=12,
    n_vehicles=4,
    n_clusters=20,
    rollout_depth=20,
    mcts_iterations=100,
    max_days=1
):
    # Group requests by date
    requests_by_date = defaultdict(list)
    for r in requests:
        requests_by_date[r.date].append(r)
    # print(requests_by_date)
    # Train the generative model once globally
    print("Training generative model...")
    gen_model = ClusteredRequestGenerator(n_clusters=n_clusters)
    gen_model.fit(requests)

    mcts_vehicles_by_date = {}
    mcts_unserved_by_date = {}

    for i, date in enumerate(sorted(requests_by_date)):
        if max_days is not None and i >= max_days:
            break
        print(f"\n Running MCTS on {date}, it takes a couple of minutes to run")
        day_requests = requests_by_date[date]
        travel_time_matrix = travel_time_matrices[date]

        vehicles = [VehicleState(i, capacity=vehicle_capacity) for i in range(n_vehicles)]
        state = MDPState(vehicles, day_requests, travel_time_matrix)

        step = 0
        while not state.is_terminal():
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                print("No legal actions available — breaking early.")
                break

            mcts = MCTS(state, model=gen_model, rollout_depth=rollout_depth)
            action = mcts.search(iterations=mcts_iterations)
            # if action is None:
            #     break
            state = state.apply_action(action)
            step += 1

        print(f"{date}: Served = {state.reward()}, Unserved = {len(state.pending_requests)}")

        mcts_vehicles_by_date[date] = state.vehicles
        mcts_unserved_by_date[date] = state.pending_requests

    return mcts_vehicles_by_date, mcts_unserved_by_date


def main():
    # 1. load data
    requests, distance_matrices, travel_time_matrices = load_all_requests("data")
    print(requests[0], len(requests))

    # 2. Run all real request distribution plots
    plot_time_window_gantt(requests, "real")
    plot_hourly_demand(requests, "real")
    plot_request_location(requests, "real")

    # 3. test generative model
    future_requests = generate_synthetic_requests_clustered(requests, n_samples=120, n_clusters=20)
    print(future_requests[0], len(future_requests))
    plot_time_window_gantt(future_requests, "sampled_RandomForest")
    plot_hourly_demand(future_requests, "sampled_RandomForest")
    plot_request_location(future_requests, "sampled_RandomForest")
    
    # # 4. test greedy
    vehicles_by_date, unserved_by_date = greedy_solve(requests, travel_time_matrices)
    for date in vehicles_by_date:
        total_assigned = sum(len(v.route) // 2 for v in vehicles_by_date[date])
        print(f"{date}: Assigned {total_assigned} requests, Unserved: {len(unserved_by_date[date])}")
    
    # # Evaluate greedy
    # requests_by_date = defaultdict(list)
    # for r in requests:
    #     requests_by_date[r.date].append(r)

    # plot_vehicle_request_map(vehicles_by_date, requests_by_date)

    # 6. test MCTS 
    vehicles_by_date, unserved_by_date = run_mcts_with_clustered_model(
        requests=requests,
        travel_time_matrices=travel_time_matrices,
        max_days=1  # Remove or change to loop over all
    )
    # plot_vehicle_request_map(vehicles_by_date, unserved_by_date)


if __name__ == "__main__":
    main()
