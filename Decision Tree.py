import numpy as np
import pandas as pd

# 1. Membuat Kelas Node (Simpul dalam Pohon)
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        # Untuk node keputusan (Decision Node)
        self.feature_index = feature_index # Index kolom yang digunakan untuk membelah
        self.threshold = threshold         # Batas nilai untuk membelah (kriteria)
        self.left = left                   # Cabang ke kiri (< threshold)
        self.right = right                 # Cabang ke kanan (>= threshold)

        # Untuk node daun (Leaf Node)
        self.value = value                 # Hasil tebakan kelas (jika ini ujung cabang)

    def is_leaf_node(self):
        return self.value is not None

# 2. Membuat Kelas Utama Decision Tree
class DecisionTreeFromScratch:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split # Syarat minimal baris untuk dibagi lagi
        self.max_depth = max_depth                 # Kedalaman maksimal pohon
        self.root = None                           # Akar pohon

    def fit(self, X, y):
        # X dan y harus berupa numpy array agar mudah diiris
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_np = y.values if isinstance(y, pd.Series) else np.array(y)
        self.root = self._grow_tree(X_np, y_np)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Kondisi berhenti (Stoping Criteria):
        # Jika mencapai max_depth, label sisa 1 jenis, atau data terlalu sedikit
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Cari pemisahan (split) terbaik
        best_feat, best_thresh = self._best_split(X, y, n_features)

        # Jika tidak menemukan pemisahan yang menguntungkan, jadikan Leaf Node
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Bagi data menjadi cabang kiri dan kanan berdasarkan pemisahan terbaik
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # Rekursif: Tumbuhkan anak pohon kiri dan kanan
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left_child, right_child)

    def _best_split(self, X, y, n_features):
        best_gini = 1.0 # Gini terburuk adalah 1 (atau 0.5)
        split_idx, split_thresh = None, None

        # Evaluasi setiap fitur (kolom) dan setiap kemungkinan nilainya
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) # Coba potong di setiap nilai unik

            for thr in thresholds:
                # Menghitung Gini Impurity dari pemisahan ini
                gini = self._gini_impurity_split(y, X_column, thr)

                # Jika Gini lebih rendah (lebih murni), simpan sebagai best split
                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _gini_impurity_split(self, y, X_column, split_thresh):
        # Bagi indeks data menjadi yang masuk ke kiri dan ke kanan
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 1.0 # Split yang sangat buruk (semua numpuk di 1 sisi)

        # Hitung jumlah baris
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        # Hitung Gini dari masing-masing cabang
        gini_l = self._gini(y[left_idxs])
        gini_r = self._gini(y[right_idxs])

        # Hitung rata-rata tertimbang Gini dari split ini
        child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r
        return child_gini

    def _gini(self, y):
        # Gini = 1 - sum(p_i ^ 2)
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        # Cari mayoritas kelas (Voting)
        y = list(y)
        return max(set(y), key=y.count)

    def predict(self, X):
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X_np])

    def _traverse_tree(self, x, node):
        # Jika sampai di daun, kembalikan nilai tebakannya
        if node.is_leaf_node():
            return node.value

        # Menentukan apakah masuk ke cabang kiri atau kanan
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)




# 2. Buat model dari "sketsasketsa'
model_manual = DecisionTreeFromScratch(max_depth=5, min_samples_split=2)

# 3. Latih Model (Model akan mencari split terbaik secara rekursif)
model_manual.fit(X_train, y_train)

# 4. Lakukan Prediksi
y_pred_manual = model_manual.predict(X_test)

# 5. Hitung Akurasi (Manual)
benar = np.sum(y_pred_manual == np.array(y_test))
total = len(y_test)
akurasi = (benar / total) * 100

print(f"Akurasi Model Manual: {akurasi:.2f}%")