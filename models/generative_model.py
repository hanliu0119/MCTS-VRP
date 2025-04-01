import numpy as np
from math import sin, cos, pi
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Encodes time of day as sine and cosine values for cyclic representation
def encode_time_of_day(seconds):
    t = seconds / 86400
    return sin(2 * pi * t), cos(2 * pi * t)


class ClusteredRequestGenerator:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = None
        self.classifier = None

    # Trains KMeans for dropoff clustering and RandomForest to predict clusters from pickup info
    def fit(self, requests):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        dropoffs = np.array([r.dropoff_loc for r in requests])
        self.kmeans.fit(dropoffs)

        X, y = self._prepare_features_labels(requests)
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)

        train_preds = self.classifier.predict(X_train)
        val_preds = self.classifier.predict(X_val)
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)

        print(f"Random Forest: Prediction Accuracy:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val   Accuracy: {val_acc:.4f}")

    # Predicts the dropoff cluster given pickup location and time
    def predict_cluster(self, pickup_lat, pickup_lon, pickup_time):
        sin_t, cos_t = encode_time_of_day(pickup_time)
        X_input = [[pickup_lat, pickup_lon, sin_t, cos_t]]
        X_scaled = self.scaler.transform(X_input)
        cluster = self.classifier.predict(X_scaled)[0]
        return cluster

    # Returns the center of the given dropoff cluster
    def get_dropoff_center(self, cluster_id):
        return self.kmeans.cluster_centers_[cluster_id]
    
    # Prepares features and corresponding dropoff cluster labels for training
    def _prepare_features_labels(self, requests):
        X, y = [], []
        for r in requests:
            plat, plon = r.pickup_loc
            sin_t, cos_t = encode_time_of_day(r.pickup_tw[0])
            X.append([plat, plon, sin_t, cos_t])
            label = self.kmeans.predict([r.dropoff_loc])[0]
            y.append(label)
        return np.array(X), np.array(y)
