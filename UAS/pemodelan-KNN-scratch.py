import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


class KNNClassifierScratch:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
    def predict(self, X, batch_size=128):
        """Predict labels for X using batched distance computation to save memory.

        batch_size: number of test samples processed per batch.
        """
        X = np.asarray(X, dtype=float)
        n_test = X.shape[0]
        y_pred = np.empty(n_test, dtype=self.y_train.dtype)

        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            Xb = X[start:end]
            # compute distances between Xb and all train samples: shape (b, n_train)
            dists = np.sqrt(((Xb[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2))

            for i in range(dists.shape[0]):
                di = dists[i]
                idx = np.argsort(di)[:self.n_neighbors]
                neigh_labels = self.y_train[idx]
                if self.weights == 'distance':
                    neigh_dists = di[idx]
                    neigh_dists = np.where(neigh_dists == 0, 1e-9, neigh_dists)
                    unique_labels = np.unique(neigh_labels)
                    votes = {}
                    for lab in unique_labels:
                        mask = (neigh_labels == lab)
                        votes[lab] = np.sum(1.0 / neigh_dists[mask])
                    y_pred[start + i] = max(votes.items(), key=lambda x: (x[1], -x[0]))[0]
                else:
                    vals, counts = np.unique(neigh_labels, return_counts=True)
                    max_count = counts.max()
                    candidates = vals[counts == max_count]
                    if len(candidates) == 1:
                        y_pred[start + i] = candidates[0]
                    else:
                        best = None
                        best_avg = float('inf')
                        for c in candidates:
                            mask = (neigh_labels == c)
                            avgd = np.mean(di[idx][mask])
                            if avgd < best_avg:
                                best_avg = avgd
                                best = c
                        y_pred[start + i] = best

        return y_pred


def main():
    try:
        df = pd.read_csv('dataset_siap_model_final.csv', index_col='ID_Individu')
    except FileNotFoundError:
        print("Kesalahan: File 'dataset_siap_model_final.csv' tidak ditemukan.")
        return

    target_col = 'Tingkat_Kemiskinan'
    if target_col not in df.columns:
        print(f"Kesalahan: Kolom {target_col} tidak ditemukan.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    print("Melatih KNN (scratch) pada data hasil SMOTE...")
    knn_scratch = KNNClassifierScratch(n_neighbors=5, weights='distance')
    knn_scratch.fit(X_train_scaled, y_train_smote)

    y_pred = knn_scratch.predict(X_test_scaled)

    print("\n--- EVALUASI KNN (SCRATCH) ---")
    print(f"Akurasi Keseluruhan: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)

if __name__ == '__main__':
    main()
