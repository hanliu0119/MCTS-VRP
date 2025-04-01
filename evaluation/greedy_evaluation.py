import matplotlib.pyplot as plt
import os

# Plots and saves the routes taken by each vehicle on a map for each date
def plot_vehicle_request_map(vehicles_by_date, requests_by_date, save_folder="greedy_maps"):
    os.makedirs(save_folder, exist_ok=True)

    for date, vehicles in vehicles_by_date.items():
        reqs = {r.pickup_id: r for r in requests_by_date[date]}
        reqs.update({r.dropoff_id: r for r in requests_by_date[date]})

        plt.figure(figsize=(10, 8))
        colors = ["red", "blue", "green", "orange"]

        for i, v in enumerate(vehicles):
            color = colors[i % len(colors)]
            lats, lons = [], []

            for rid in v.route:
                if rid in reqs:
                    lat, lon = reqs[rid].pickup_loc if rid == reqs[rid].pickup_id else reqs[rid].dropoff_loc
                    lats.append(lat)
                    lons.append(lon)

            plt.plot(lons, lats, marker='o', linestyle='-', color=color, label=f"Vehicle {v.id}")

        plt.title(f"Greedy Route Map — {date}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./outputs/greedy_map_{date}.png")
        plt.close()

    print(f" Saved greedy route maps to '{save_folder}'")
