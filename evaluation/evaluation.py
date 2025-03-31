import random
import numpy as np
import matplotlib.pyplot as plt
from models.generative_model import sample_future_requests


def evaluate_model_fit(model, requests, n=500):
    real_dropoffs = []
    pred_dropoffs = []

    model.eval()
    for req in random.sample(requests, min(n, len(requests))):
        pickup_lat, pickup_lon = req.pickup_loc
        pickup_time = req.pickup_tw[0]
        real_dropoffs.append(req.dropoff_loc)

        input_tensor = model.model(torch.tensor([[pickup_lat, pickup_lon, pickup_time]], dtype=torch.float32))
        pred = input_tensor.squeeze().detach().numpy()
        pred_dropoffs.append((pred[0], pred[1]))

    real_dropoffs = np.array(real_dropoffs)
    pred_dropoffs = np.array(pred_dropoffs)

    plt.figure(figsize=(8, 6))
    plt.scatter(real_dropoffs[:, 0], real_dropoffs[:, 1], alpha=0.5, label="Real Dropoffs")
    plt.scatter(pred_dropoffs[:, 0], pred_dropoffs[:, 1], alpha=0.5, label="Predicted Dropoffs")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.legend()
    plt.title("Model Fit: Real vs. Predicted Dropoff Locations")
    plt.grid(True)
    plt.savefig("outputs/model_fit_comparison.png")
    plt.show()


def evaluate_sampled_requests(model, requests, n=500):
    real_dropoffs = [req.dropoff_loc for req in random.sample(requests, min(n, len(requests)))]
    sampled = sample_future_requests(model, requests, n)
    pred_dropoffs = [s["dropoff"][:2] for s in sampled]

    real_dropoffs = np.array(real_dropoffs)
    pred_dropoffs = np.array(pred_dropoffs)

    plt.figure(figsize=(8, 6))
    plt.scatter(real_dropoffs[:, 0], real_dropoffs[:, 1], alpha=0.5, label="Real Dropoffs")
    plt.scatter(pred_dropoffs[:, 0], pred_dropoffs[:, 1], alpha=0.5, label="Sampled Dropoffs")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.legend()
    plt.title("Sampled Requests: Real vs. Generated Dropoff Locations")
    plt.grid(True)
    plt.savefig("outputs/sampled_requests_comparison.png")
    plt.show()


def plot_real_requests(requests, n=500):
    sampled_reqs = random.sample(requests, min(n, len(requests)))
    dropoffs = np.array([req.dropoff_loc for req in sampled_reqs])
    times = np.array([req.dropoff_tw[0] for req in sampled_reqs])
    demands = np.array([req.demand for req in sampled_reqs])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(dropoffs[:, 0], dropoffs[:, 1], c=times, s=20 + 20 * demands, alpha=0.6, cmap="viridis")
    plt.colorbar(scatter, label="Dropoff Time (s)")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title("Real Dropoff Locations (Color: Time, Size: Demand)")
    plt.grid(True)
    plt.savefig("outputs/real_dropoffs.png")
    plt.show()
