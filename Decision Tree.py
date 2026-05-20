import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# BAGIAN 1: MEMASUKKAN DAN MENYIAPKAN DATASET
# ==========================================
# 1. Membaca dataset
df = pd.read_csv("Document from zeee.csv")

# 2. Pra-pemrosesan Data (Mengisi nilai kosong dan mengubah teks ke angka)
df = df.fillna(df.median(numeric_only=True))
df = pd.get_dummies(df, drop_first=True)

# 3. Seleksi Fitur menggunakan Threshold Korelasi
# Mengambil variabel yang memiliki korelasi di atas nilai tertentu
korelasi = df.corr()['Tingkat_Kemiskinan'].abs()
threshold_value = 0.05  # Anda bisa menyesuaikan ambang batas ini
kolom_terpilih = korelasi[korelasi > threshold_value].index.tolist()

if 'Tingkat_Kemiskinan' in kolom_terpilih:
    kolom_terpilih.remove('Tingkat_Kemiskinan')

# 4. Menentukan X (Fitur) dan y (Target)
X = df[kolom_terpilih].values
y = df['Tingkat_Kemiskinan'].values

# 5. Membagi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ==========================================
# BAGIAN 2: KELAS DECISION TREE (SKETSA)
# ==========================================
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index 
        self.threshold = threshold         
        self.left = left                   
        self.right = right                 
        self.value = value                 

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeFromScratch:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth                 
        self.root = None                           

    def fit(self, X, y):
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_np = y.values if isinstance(y, pd.Series) else np.array(y)
        self.root = self._grow_tree(X_np, y_np)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y, n_features)

        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left_child, right_child)

    def _best_split(self, X, y, n_features):
        best_gini = 1.0 
        split_idx, split_thresh = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) 

            for thr in thresholds:
                gini = self._gini_impurity_split(y, X_column, thr)
                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _gini_impurity_split(self, y, X_column, split_thresh):
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 1.0 

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_l = self._gini(y[left_idxs])
        gini_r = self._gini(y[right_idxs])
        child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r
        return child_gini

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        y = list(y)
        return max(set(y), key=y.count)

    def predict(self, X):
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X_np])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# ==========================================
# BAGIAN 3: MELATIH DAN MENGUJI MODEL
# ==========================================
# Buat model 
model_manual = DecisionTreeFromScratch(max_depth=5, min_samples_split=2)

print("Sedang melatih model Decision Tree manual...")
# Latih Model 
model_manual.fit(X_train, y_train)

# Lakukan Prediksi
y_pred_manual = model_manual.predict(X_test)

# Hitung Akurasi 
benar = np.sum(y_pred_manual == np.array(y_test))
total = len(y_test)
akurasi = (benar / total) * 100

print(f"Akurasi Model Manual: {akurasi:.2f}%")
