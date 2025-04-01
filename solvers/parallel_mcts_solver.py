# from concurrent.futures import ThreadPoolExecutor, as_completed
# from collections import defaultdict
# from models.generative_model import ClusteredRequestGenerator
# from models.mdp import VehicleState, MDPState
# from solvers.mcts_solver import MCTS


# def run_mcts_with_clustered_model(
#     requests,
#     travel_time_matrices,
#     vehicle_capacity=12,
#     n_vehicles=4,
#     n_clusters=20,
#     rollout_depth=20,
#     mcts_iterations=100,
#     max_days=11,
#     num_threads=11
# ):
#     print("Training generative model...")
#     gen_model = ClusteredRequestGenerator(n_clusters=n_clusters)
#     gen_model.fit(requests)

#     # Group requests by date
#     requests_by_date = defaultdict(list)
#     for r in requests:
#         requests_by_date[r.date].append(r)

#     # Filter top dates (respecting max_days)
#     sorted_dates = sorted(requests_by_date)[:max_days]

#     # Result containers
#     mcts_vehicles_by_date = {}
#     mcts_unserved_by_date = {}

#     def run_mcts_for_date(date):
#         day_requests = requests_by_date[date]
#         travel_time_matrix = travel_time_matrices[date]
#         vehicles = [VehicleState(i, capacity=vehicle_capacity) for i in range(n_vehicles)]
#         state = MDPState(vehicles, day_requests, travel_time_matrix)

#         step = 0
#         while not state.is_terminal():
#             legal_actions = state.get_legal_actions()
#             if not legal_actions:
#                 print(f"{date} No legal actions — breaking early.")
#                 break

#             mcts = MCTS(state, model=gen_model, rollout_depth=rollout_depth)
#             action = mcts.search(iterations=mcts_iterations)
#             if action is None:
#                 print(f"{date} MCTS returned no action — breaking early.")
#                 break

#             state = state.apply_action(action)
#             step += 1

#         print(f"{date}: Served = {state.reward()}, Unserved = {len(state.pending_requests)}")
#         return date, state.vehicles, state.pending_requests

#     # Run in parallel
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         futures = [executor.submit(run_mcts_for_date, date) for date in sorted_dates]
#         for future in as_completed(futures):
#             date, vehicles, unserved = future.result()
#             mcts_vehicles_by_date[date] = vehicles
#             mcts_unserved_by_date[date] = unserved

#     return mcts_vehicles_by_date, mcts_unserved_by_date
