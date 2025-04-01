import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from math import sin, cos, pi
from collections import defaultdict

from models.data_loader import Request

# Defines a simple feedforward neural network to predict dropoff deltas and delay.
class SimpleRequestGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # dlat, dlon, delay (hours)
        )

    def forward(self, x):
        return self.model(x)

# Encodes time-of-day as sine and cosine components for cyclic representation.
def encode_time_of_day(seconds):
    t = seconds / 86400  # convert to [0, 1]
    t = max(0, min(1, t))
    return sin(2 * pi * t), cos(2 * pi * t)

# Prepares input-output training data from request objects for the neural model.
def prepare_dataset(requests):
    X, Y = [], []
    for req in requests:
        lat, lon = req.pickup_loc
        pickup_time = req.pickup_tw[0]
        sin_t, cos_t = encode_time_of_day(pickup_time)
        X.append([lat, lon, sin_t, cos_t])

        dlat = req.dropoff_loc[0] - lat
        dlon = req.dropoff_loc[1] - lon
        delay_hours = (req.dropoff_tw[0] - pickup_time) / 3600  # Normalize to hours
        Y.append([dlat, dlon, delay_hours])

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Trains the neural network model using MSE loss on delta location and delay prediction.
def train_model(requests, epochs=100, lr=1e-3):
    X, Y = prepare_dataset(requests)
    split = int(0.8 * len(X))
    train_X, val_X = X[:split], X[split:]
    train_Y, val_Y = Y[:split], Y[split:]

    model = SimpleRequestGenerator()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(train_X)
        loss = criterion(preds, train_Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(val_X)
                val_loss = criterion(val_preds, val_Y)
            # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    return model

# Samples synthetic future requests using the trained model and historical distribution.
def sample_future_requests(model, requests, n=120):
    samples = []
    model.eval()
    if not requests:
        return []

    # Get real lat/lon range
    all_lats = [r.pickup_loc[0] for r in requests] + [r.dropoff_loc[0] for r in requests]
    all_lons = [r.pickup_loc[1] for r in requests] + [r.dropoff_loc[1] for r in requests]
    lat_range = (min(all_lats), max(all_lats))
    lon_range = (min(all_lons), max(all_lons))

    # Bin requests by pickup hour to estimate time distribution
    hourly_bins = defaultdict(list)
    for r in requests:
        hour = r.pickup_tw[0] // 3600
        hourly_bins[hour].append(r)

    total = sum(len(b) for b in hourly_bins.values())
    hour_probs = [len(hourly_bins[h]) / total if h in hourly_bins else 0 for h in range(24)]
    sampled_hours = np.random.choice(range(24), size=n, p=hour_probs)

    for hour in sampled_hours:
        if not hourly_bins[hour]:
            continue
        base = random.choice(hourly_bins[hour])
        plat, plon = base.pickup_loc
        pt = base.pickup_tw[0]
        sin_t, cos_t = encode_time_of_day(pt)
        input_tensor = torch.tensor([[plat, plon, sin_t, cos_t]], dtype=torch.float32)

        with torch.no_grad():
            dlat, dlon, delay_hr = model(input_tensor).squeeze().numpy()

        # Clamp predictions to prevent out-of-bounds
        max_delta_lat = 0.02
        max_delta_lon = 0.02
        dlat = np.clip(dlat, -max_delta_lat, max_delta_lat)
        dlon = np.clip(dlon, -max_delta_lon, max_delta_lon)

        dropoff_lat = np.clip(plat + dlat, *lat_range)
        dropoff_lon = np.clip(plon + dlon, *lon_range)
        dropoff_time = pt + max(60, delay_hr * 3600)  # convert hours → seconds

        samples.append(Request(
            date="future_date",
            pickup_id=0,
            pickup_loc=(plat, plon),
            pickup_tw=(int(pt), int(pt + 1800)),
            dropoff_id=0,
            dropoff_loc=(float(dropoff_lat), float(dropoff_lon)),
            dropoff_tw=(int(dropoff_time), int(dropoff_time + 1800)),
            demand=base.demand,
            service_time=60
        ))

    return samples


# Code to train NN model to generate future request
# model, norm_stats = train_model(requests)
# future_requests = sample_future_requests(model, requests, norm_stats, n=120)
# print(future_requests[0])
# print(f"Sampled {len(future_requests)} synthetic requests for 'future_date'")

# # Evaluate generative model
# evaluate_model_fit(model, requests, norm_stats=norm_stats, threshold_meters=500)
# plot_time_window_gantt(future_requests, "sampled")
# plot_hourly_demand(future_requests, "sampled")
# plot_request_location(future_requests, "sampled")