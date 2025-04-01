import torch
import numpy as np
from models.generative_model import encode_time_of_day  # Converts time to sin/cos encoding
# from sklearn.metrics import mean_squared_error

# Evaluates a trained neural generative model on MSE and spatial accuracy
def evaluate_model_fit(model, requests, norm_stats, threshold_meters=500):

    # Computes haversine distance (in meters) between two lat/lon points
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Normalizes a value using min-max scaling
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val + 1e-8)

    # Reverses the normalization to return to original scale
    def denormalize(value, min_val, max_val):
        return value * (max_val - min_val + 1e-8) + min_val

    lat_min, lat_max = norm_stats['lat']
    lon_min, lon_max = norm_stats['lon']

    # Prepares normalized input/output tensors for evaluation
    def prepare_data(requests):
        X, Y = [], []
        for req in requests:
            plat, plon = req.pickup_loc
            dlat, dlon = req.dropoff_loc
            sin_t, cos_t = encode_time_of_day(req.pickup_tw[0])
            X.append([normalize(plat, lat_min, lat_max), normalize(plon, lon_min, lon_max), sin_t, cos_t])
            Y.append([
                normalize(dlat, lat_min, lat_max),
                normalize(dlon, lon_min, lon_max),
                (req.dropoff_tw[0] - req.pickup_tw[0]) / 3600  # Delay in hours
            ])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    # Split into training and validation sets
    X, Y = prepare_data(requests)
    split = int(0.8 * len(X))
    train_X, val_X = X[:split], X[split:]
    train_Y, val_Y = Y[:split], Y[split:]

    # Run model predictions on both sets
    model.eval()
    with torch.no_grad():
        train_pred = model(train_X).numpy()
        val_pred = model(val_X).numpy()
        train_true = train_Y.numpy()
        val_true = val_Y.numpy()

        # Computes accuracy based on spatial error within a meter threshold
        def compute_accuracy(X_input, Y_true, Y_pred):
            count = 0
            total = len(X_input)
            for x, y_t, y_p in zip(X_input, Y_true, Y_pred):
                plat, plon = denormalize(x[0], lat_min, lat_max), denormalize(x[1], lon_min, lon_max)
                t_lat, t_lon = denormalize(y_t[0], lat_min, lat_max), denormalize(y_t[1], lon_min, lon_max)
                p_lat, p_lon = denormalize(y_p[0], lat_min, lat_max), denormalize(y_p[1], lon_min, lon_max)
                dist = haversine_distance(t_lat, t_lon, p_lat, p_lon)
                if dist < threshold_meters:
                    count += 1
            return count / total

        # Compute distance-based accuracy
        train_acc = compute_accuracy(train_X.numpy(), train_true, train_pred)
        val_acc = compute_accuracy(val_X.numpy(), val_true, val_pred)

    # Output evaluation results
    print(f" Model Evaluation:")
    print(f"  Train MSE: {np.mean((train_pred - train_true)**2):.4f}")
    print(f"  Val   MSE: {np.mean((val_pred - val_true)**2):.4f}")
    print(f"  Train Accuracy (<{threshold_meters}m): {train_acc * 100:.2f}%")
    print(f"  Val   Accuracy (<{threshold_meters}m): {val_acc * 100:.2f}%")
