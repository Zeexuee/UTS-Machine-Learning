import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


class LinearSVM_OVR:
    """Simple linear SVM One-vs-Rest trained with SGD on hinge loss.
    This is a lightweight educational implementation (not optimized).
    """
    def __init__(self, lr=0.01, epochs=100, C=1.0, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.verbose = verbose

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # weight matrix: one weight vector per class
        self.W = np.zeros((n_classes, n_features), dtype=float)
        self.b = np.zeros(n_classes, dtype=float)

        # train one-vs-rest
        for idx, cls in enumerate(self.classes_):
            y_binary = np.where(y == cls, 1, -1)
            w = np.zeros(n_features, dtype=float)
            b = 0.0

            for ep in range(self.epochs):
                # simple SGD loop
                for i in range(X.shape[0]):
                    xi = X[i]
                    yi = y_binary[i]
                    margin = yi * (np.dot(w, xi) + b)
                    if margin >= 1:
                        # subgradient for hinge when correct with margin
                        grad_w = w
                        # bias grad is zero
                    else:
                        # violated margin
                        grad_w = w - self.C * yi * xi
                        # bias subgradient
                        b += self.lr * self.C * yi

                    # gradient descent step
                    w -= self.lr * grad_w

                if self.verbose and (ep % 10 == 0):
                    # compute hinge loss (objective approx)
                    margins = y_binary * (X.dot(w) + b)
                    loss = 0.5 * np.dot(w, w) + self.C * np.sum(np.maximum(0, 1 - margins))
                    print(f"Class {cls} epoch {ep} loss {loss:.4f}")

            self.W[idx, :] = w
            self.b[idx] = b

    def decision_function(self, X):
        return X.dot(self.W.T) + self.b

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]


def main():
    # 1. Memuat Dataset
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

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. SMOTE pada data latih
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    # 5. Inisialisasi dan pelatihan SVM scratch
    print("Melatih Linear SVM (scratch, OVR) pada data hasil SMOTE...")
    svm_scratch = LinearSVM_OVR(lr=0.001, epochs=100, C=1.0, verbose=False)
    svm_scratch.fit(X_train_scaled, y_train_smote)

    # 6. Prediksi dan evaluasi
    y_pred = svm_scratch.predict(X_test_scaled)

    print("\n--- EVALUASI SVM (SCRATCH) ---")
    print(f"Akurasi Keseluruhan: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)

if __name__ == '__main__':
    main()
