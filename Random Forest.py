import csv
import random
import math
from collections import Counter

# ==========================================
# 1. FUNGSI PEMBANTU (MATEMATIKA)
# ==========================================
def hitung_gini(y):
    """Menghitung Gini Impurity untuk mengukur seberapa 'bercampur' kelas dalam node."""
    if not y: return 0
    counts = Counter(y)
    impurity = 1.0
    for jumlah in counts.values():
        probabilitas = jumlah / len(y)
        impurity -= probabilitas ** 2
    return impurity

def bagi_data(X, y, indeks_fitur, threshold):
    """Membagi data menjadi dua cabang berdasarkan nilai batas (threshold)."""
    X_kiri, y_kiri, X_kanan, y_kanan = [], [], [], []
    for i in range(len(X)):
        if X[i][indeks_fitur] <= threshold:
            X_kiri.append(X[i])
            y_kiri.append(y[i])
        else:
            X_kanan.append(X[i])
            y_kanan.append(y[i])
    return X_kiri, y_kiri, X_kanan, y_kanan

# ==========================================
# 2. KOMPONEN DECISION TREE (POHON KEPUTUSAN)
# ==========================================
class Node:
    """Representasi satu titik percabangan atau daun dalam pohon."""
    def __init__(self, indeks_fitur=None, threshold=None, kiri=None, kanan=None, *, nilai_daun=None):
        self.indeks_fitur = indeks_fitur
        self.threshold = threshold
        self.kiri = kiri
        self.kanan = kanan
        self.nilai_daun = nilai_daun # Hanya memiliki isi jika node ini adalah hasil akhir (daun)

    def is_leaf(self):
        return self.nilai_daun is not None

class DecisionTree:
    """Algoritma membangun satu buah pohon keputusan."""
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.akar = None

    def fit(self, X, y):
        # Kunci Random Forest: Pohon tidak melihat semua kolom. 
        # Secara default, pohon hanya melihat akar kuadrat dari total jumlah kolom.
        self.n_features = self.n_features if self.n_features else int(math.sqrt(len(X[0])))
        self.akar = self._bangun_tree(X, y, kedalaman=0)

    def _bangun_tree(self, X, y, kedalaman):
        n_samples, n_total_features = len(X), len(X[0])
        n_kelas_unik = len(set(y))

        # Kriteria Berhenti: Jika pohon terlalu dalam, atau data sudah 1 kelas semua
        if (kedalaman >= self.max_depth or n_kelas_unik == 1 or n_samples < self.min_samples_split):
            daun_value = self._nilai_terbanyak(y)
            return Node(nilai_daun=daun_value)

        # Kunci Random Forest: Pilih fitur (kolom) secara acak untuk percabangan ini
        fitur_acak = random.sample(range(n_total_features), self.n_features)
        
        split_terbaik = self._cari_split_terbaik(X, y, fitur_acak)
        
        if not split_terbaik:
            return Node(nilai_daun=self._nilai_terbanyak(y))

        cabang_kiri = self._bangun_tree(split_terbaik['X_kiri'], split_terbaik['y_kiri'], kedalaman + 1)
        cabang_kanan = self._bangun_tree(split_terbaik['X_kanan'], split_terbaik['y_kanan'], kedalaman + 1)

        return Node(indeks_fitur=split_terbaik['indeks_fitur'], threshold=split_terbaik['threshold'], 
                    kiri=cabang_kiri, kanan=cabang_kanan)

    def _cari_split_terbaik(self, X, y, fitur_acak):
        gini_terbaik = 999
        split_terbaik = {}
        
        for indeks_fitur in fitur_acak:
            # Cari threshold (batas nilai) dari semua baris di fitur tertentu
            nilai_fitur = [baris[indeks_fitur] for baris in X]
            thresholds = set(nilai_fitur)
            
            for threshold in thresholds:
                X_kiri, y_kiri, X_kanan, y_kanan = bagi_data(X, y, indeks_fitur, threshold)
                if not y_kiri or not y_kanan:
                    continue
                
                # Evaluasi sebaik apa pemisahan ini membedakan tingkat kemiskinan
                n = len(y)
                gini_gabungan = (len(y_kiri)/n) * hitung_gini(y_kiri) + (len(y_kanan)/n) * hitung_gini(y_kanan)
                
                if gini_gabungan < gini_terbaik:
                    gini_terbaik = gini_gabungan
                    split_terbaik = {
                        'indeks_fitur': indeks_fitur, 'threshold': threshold,
                        'X_kiri': X_kiri, 'y_kiri': y_kiri, 'X_kanan': X_kanan, 'y_kanan': y_kanan
                    }
        return split_terbaik

    def _nilai_terbanyak(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return [self._prediksi_satu(x, self.akar) for x in X]

    def _prediksi_satu(self, x, node):
        if node.is_leaf(): return node.nilai_daun
        if x[node.indeks_fitur] <= node.threshold:
            return self._prediksi_satu(x, node.kiri)
        return self._prediksi_satu(x, node.kanan)


# ==========================================
# 3. KOMPONEN RANDOM FOREST
# ==========================================
class RandomForestManual:
    """Menggabungkan banyak Decision Tree menjadi satu hutan."""
    def __init__(self, n_trees=5, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            print(f"Membangun Pohon ke-{i+1}...")
            # Bootstrap Aggregating (Bagging): Ambil sampel baris data secara acak 
            # (boleh ada data duplikat/dengan pengembalian)
            X_sampel, y_sampel = self._bootstrap_sample(X, y)
            
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sampel, y_sampel)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = len(X)
        X_sampel, y_sampel = [], []
        for _ in range(n_samples):
            indeks_acak = random.randint(0, n_samples - 1)
            X_sampel.append(X[indeks_acak])
            y_sampel.append(y[indeks_acak])
        return X_sampel, y_sampel

    def predict(self, X):
        # Minta setiap pohon menebak tingkat kemiskinan
        prediksi_pohon = [tree.predict(X) for tree in self.trees]
        # Ubah orientasi matriks agar mudah dihitung votingnya
        prediksi_pohon_t = list(map(list, zip(*prediksi_pohon)))
        # Majority Voting: Pilih tebakan yang paling banyak di-voting oleh pohon-pohon
        prediksi_final = [Counter(baris).most_common(1)[0][0] for baris in prediksi_pohon_t]
        return prediksi_final


# ==========================================
# 4. EKSEKUSI PADA DATASET
# ==========================================
if __name__ == "__main__":
    dataset_X = []
    dataset_y = []

    print("Membaca dataset 'Document from zeee.csv'...")
    with open('Document from zeee.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        
        # Pisahkan Tingkat Kemiskinan sebagai target (Y) dan sisanya sebagai fitur (X)
        indeks_target = header.index('Tingkat_Kemiskinan')
        
        for baris in reader:
            baris_fitur = []
            for i in range(len(baris)):
                if i == indeks_target:
                    dataset_y.append(int(float(baris[i])))
                else:
                    baris_fitur.append(float(baris[i]))
            dataset_X.append(baris_fitur)

    # Memisahkan 200 baris terakhir khusus untuk pengetesan (Testing Data)
    X_train, y_train = dataset_X[:-200], dataset_y[:-200]
    X_test, y_test = dataset_X[-200:], dataset_y[-200:]

    # Karena komputasi murni Python cukup berat untuk >9.000 baris,
    # kita gunakan 5 pohon (n_trees=5) dengan kedalaman maksimal 8.
    print(f"Melatih Random Forest dengan {len(X_train)} baris data latih...")
    rf_model = RandomForestManual(n_trees=5, max_depth=8)
    rf_model.fit(X_train, y_train)

    print("Melakukan prediksi pada 200 data uji (Testing)...")
    prediksi = rf_model.predict(X_test)

    # Hitung Akurasi
    benar = sum(1 for p, a in zip(prediksi, y_test) if p == a)
    akurasi = (benar / len(y_test)) * 100
    print(f"\nAkurasi Prediksi Model: {akurasi:.2f}%")
