import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
import os

def seconds_to_time(seconds):
    total_minutes = (seconds // 60)
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"

def plot_request_location(requests, label):
    os.makedirs("outputs", exist_ok=True)
    dropoffs = np.array([req.dropoff_loc for req in requests])
    demands = np.array([req.demand for req in requests])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        dropoffs[:, 0],
        dropoffs[:, 1],
        c=demands,
        s=40,
        alpha=0.5,
        cmap="plasma"
    )
    plt.colorbar(scatter, label="Demand")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title(f"Demands Location Map ({label})")
    plt.grid(True)
    plt.savefig(f"outputs/{label}_demand_locations.png")
    plt.close()

def plot_hourly_demand(requests, label):
    os.makedirs("outputs", exist_ok=True)
    times = [req.pickup_tw[0] for req in requests]
    demands = [req.demand for req in requests]
    dates = [req.date for req in requests]

    df = pd.DataFrame({'date': dates, 'time': times, 'demand': demands})
    df['hour'] = df['time'] // 3600
    hourly_by_day = df.groupby(['date', 'hour'])['demand'].sum().reset_index()
    average_hourly = hourly_by_day.groupby('hour')['demand'].mean()

    hour_labels = [seconds_to_time(h * 3600) for h in average_hourly.index]

    plt.figure(figsize=(12, 4))
    plt.bar(hour_labels, average_hourly.values, color='skyblue', edgecolor='black')
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Demand")
    plt.title(f"Average Hourly Demand Across All Dates ({label})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"outputs/{label}_hourly_demand_curve.png")
    plt.close()

def plot_time_window_gantt(requests, label):
    output_dir = f"outputs/Gantt_chart_request_{label}"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame([{
        'date': getattr(req, 'date', 'unknown'),
        'pickup_start': req.pickup_tw[0],
        'pickup_end': req.pickup_tw[1],
        'dropoff_start': req.dropoff_tw[0],
        'dropoff_end': req.dropoff_tw[1]
    } for req in requests])

    for date, group in df.groupby("date"):
        plt.figure(figsize=(12, 6))
        group = group.reset_index(drop=True)

        for i, row in group.iterrows():
            p_start_hr = row['pickup_start'] / 3600
            p_end_hr = row['pickup_end'] / 3600
            d_start_hr = row['dropoff_start'] / 3600
            d_end_hr = row['dropoff_end'] / 3600

            plt.hlines(i, p_start_hr, p_end_hr, colors='blue', label='Pickup' if i == 0 else "")
            plt.hlines(i, d_start_hr, d_end_hr, colors='green', label='Dropoff' if i == 0 else "")

        plt.xlabel("Hour of Day")
        plt.ylabel("Request Index")
        plt.title(f"Time Windows Gantt Chart — {date} ({label})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filepath = f"{output_dir}/{date}_Gantt_chart.png"
        plt.savefig(filepath)
        plt.close()

    print(f"✅ Gantt charts saved to: {output_dir}")
