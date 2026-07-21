import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


class SoftmaxRegression:
    """Multinomial Logistic Regression (softmax) trained with mini-batch SGD.
    Vectorized implementation with L2 regularization.
    """
    def __init__(self, lr=0.1, epochs=500, batch_size=64, reg_lambda=0.01, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.verbose = verbose

    def _one_hot(self, y):
        classes, y_idx = np.unique(y, return_inverse=True)
        Y = np.zeros((y.shape[0], classes.shape[0]))
        Y[np.arange(y.shape[0]), y_idx] = 1
        return Y, classes

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        Y, classes = self._one_hot(y)
        self.classes_ = classes
        n_classes = Y.shape[1]

        # initialize weights
        self.W = np.zeros((n_features, n_classes), dtype=float)
        self.b = np.zeros(n_classes, dtype=float)

        for epoch in range(self.epochs):
            # shuffle
            idx = np.random.permutation(n_samples)
            X_shuf = X[idx]
            Y_shuf = Y[idx]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                xb = X_shuf[start:end]
                yb = Y_shuf[start:end]

                # forward
                logits = xb.dot(self.W) + self.b
                # numeric stability
                logits -= np.max(logits, axis=1, keepdims=True)
                exp_scores = np.exp(logits)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                # gradient
                m = xb.shape[0]
                grad_logits = (probs - yb) / m
                grad_W = xb.T.dot(grad_logits) + self.reg_lambda * self.W
                grad_b = np.sum(grad_logits, axis=0)

                # update
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

            if self.verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                loss = self._loss(X, Y)
                print(f"Epoch {epoch+1}/{self.epochs} loss={loss:.4f}")

    def _loss(self, X, Y):
        logits = X.dot(self.W) + self.b
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        core = -np.sum(Y * np.log(probs + 1e-15)) / X.shape[0]
        reg = 0.5 * self.reg_lambda * np.sum(self.W * self.W)
        return core + reg

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        logits = X.dot(self.W) + self.b
        return self.classes_[np.argmax(logits, axis=1)]


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

    # 5. Inisialisasi dan pelatihan Logistic Regression scratch
    print("Melatih Logistic Regression (scratch, softmax) pada data hasil SMOTE...")
    lr_scratch = SoftmaxRegression(lr=0.1, epochs=500, batch_size=64, reg_lambda=0.01, verbose=True)
    lr_scratch.fit(X_train_scaled, y_train_smote)

    # 6. Prediksi dan evaluasi
    y_pred = lr_scratch.predict(X_test_scaled)

    print("\n--- EVALUASI LOGISTIC REGRESSION (SCRATCH) ---")
    print(f"Akurasi Keseluruhan: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)

if __name__ == '__main__':
    main()
