import math
import random
import csv

# 1. Fungsi untuk menghitung jarak (Euclidean Distance)
def hitung_jarak(titik1, titik2):
    total_kuadrat = 0
    # Mengurangi setiap dimensi fitur, mengkuadratkan, lalu dijumlahkan
    for i in range(len(titik1)):
        total_kuadrat += (titik1[i] - titik2[i]) ** 2
    return math.sqrt(total_kuadrat)

# 2. Fungsi Utama K-Means
def k_means_manual(data, k, maksimal_iterasi=100):
    # a. Inisialisasi: Pilih k titik secara acak dari data sebagai pusat awal (centroid)
    centroids = random.sample(data, k)

    for iterasi in range(maksimal_iterasi):
        # Siapkan wadah kosong untuk cluster
        clusters = [[] for _ in range(k)]

        # b. Penugasan Data (Assignment): Kelompokkan setiap titik data ke centroid terdekat
        for titik in data:
            jarak_ke_centroids = [hitung_jarak(titik, c) for c in centroids]
            index_terdekat = jarak_ke_centroids.index(min(jarak_ke_centroids))
            clusters[index_terdekat].append(titik)

        # c. Pembaruan Centroid (Update): Hitung rata-rata posisi dari titik-titik di setiap cluster
        centroid_baru = []
        for cluster in clusters:
            if not cluster: # Jika ada cluster yang kebetulan kosong
                centroid_baru.append(random.choice(data))
                continue

            jumlah_dimensi = len(data[0])
            rata_rata_dimensi = []

            for d in range(jumlah_dimensi):
                total_nilai = sum(titik[d] for titik in cluster)
                rata_rata_dimensi.append(total_nilai / len(cluster))

            centroid_baru.append(rata_rata_dimensi)

        # d. Cek Konvergensi: Jika centroid tidak berubah lagi, hentikan perulangan
        if centroid_baru == centroids:
            print(f"Konvergensi tercapai pada iterasi ke-{iterasi+1}")
            break

        centroids = centroid_baru

    return clusters, centroids

# ==========================================

# ==========================================

# Membaca dataset CSV secara manual
dataset = []
with open('Document from zeee.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader) # Melewati baris pertama (nama kolom)

    # Mencari indeks kolom 'Tingkat_Kemiskinan' agar tidak ikut dihitung jaraknya
    indeks_target = header.index('Tingkat_Kemiskinan')

    for baris in reader:
        # Mengubah data string menjadi float (desimal) untuk kalkulasi matematis
        # Dan mengabaikan kolom Tingkat_Kemiskinan
        baris_angka = [float(baris[i]) for i in range(len(baris)) if i != indeks_target]
        dataset.append(baris_angka)

# Karena kita tidak memakai scaler otomatis, pada data asli, kolom seperti
# 'Sewa_Rumah_Bulanan' akan mendominasi jarak. Untuk sketsa algoritma,
# kita langsung jalankan dengan K=4 pada data mentah.

print("Memulai clustering pada 9557 baris data...")
hasil_clusters, hasil_centroids = k_means_manual(dataset, k=4, maksimal_iterasi=50)

# Menampilkan Ringkasan Hasil
for i, cluster in enumerate(hasil_clusters):
    print(f"Cluster {i} memiliki {len(cluster)} anggota data.")