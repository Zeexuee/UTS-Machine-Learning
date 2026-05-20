import csv
import math
import random

# ==========================================
# 1. FUNGSI MATEMATIKA DASAR
# ==========================================
def hitung_rata_rata(angka):
    return sum(angka) / float(len(angka))

def hitung_standar_deviasi(angka):
    rata_rata = hitung_rata_rata(angka)
    # Menghitung varians
    varians = sum([(x - rata_rata)**2 for x in angka]) / float(len(angka) - 1)
    # Tambahkan angka yang sangat kecil (1e-9) untuk mencegah error dibagi nol
    return math.sqrt(varians) if varians > 0 else 1e-9

def hitung_korelasi(x, y):
    # Menghitung korelasi Pearson secara manual
    n = len(x)
    rata_x = hitung_rata_rata(x)
    rata_y = hitung_rata_rata(y)

    pembilang = sum((x[i] - rata_x) * (y[i] - rata_y) for i in range(n))
    penyebut_x = sum((x[i] - rata_x)**2 for i in range(n))
    penyebut_y = sum((y[i] - rata_y)**2 for i in range(n))

    penyebut = math.sqrt(penyebut_x * penyebut_y)
    if penyebut == 0:
        return 0
    return pembilang / penyebut

# ==========================================
# 2. MEMBACA & MENYARING DATA
# ==========================================
# 
nama_file = 'Document from zeee.csv'
dataset = []
kolom_header = []

# Membaca file CSV
with open(nama_file, 'r') as file:
    csv_reader = csv.reader(file)
    kolom_header = next(csv_reader)
    for baris in csv_reader:
        if not baris:
            continue
        # Mengubah data yang asalnya teks menjadi angka desimal (float)
        dataset.append([float(x) for x in baris])

indeks_target = kolom_header.index('Tingkat_Kemiskinan')
nilai_target = [baris[indeks_target] for baris in dataset]

# Logika Filter Threshold: Menyaring fitur dengan korelasi di atas 0.1
nilai_ambang = 0.1
indeks_terpilih = []

print("Menghitung korelasi antar fitur untuk seleksi...")
for i in range(len(kolom_header)):
    if i == indeks_target:
        indeks_terpilih.append(i) # Kolom target pasti dimasukkan
        continue

    nilai_fitur = [baris[i] for baris in dataset]
    korelasi = abs(hitung_korelasi(nilai_fitur, nilai_target))

    if korelasi > nilai_ambang:
        indeks_terpilih.append(i)

jumlah_fitur_awal = len(kolom_header) - 1
jumlah_fitur_lolos = len(indeks_terpilih) - 1
print(f"Jumlah fitur yang lolos seleksi: {jumlah_fitur_lolos} dari {jumlah_fitur_awal} fitur awal.\n")

# Membuat dataset baru hanya dengan fitur yang lolos seleksi
dataset_bersih = [[baris[i] for i in indeks_terpilih] for baris in dataset]

# ==========================================
# 3. MEMBAGI DATA LATIH & DATA UJI
# ==========================================
def bagi_data(data, rasio_uji=0.2, seed=42):
    random.seed(seed)
    data_salinan = list(data)
    random.shuffle(data_salinan) # Mengacak data

    ukuran_uji = int(len(data_salinan) * rasio_uji)
    data_uji = data_salinan[:ukuran_uji]
    data_latih = data_salinan[ukuran_uji:]
    return data_latih, data_uji

data_latih, data_uji = bagi_data(dataset_bersih)

# ==========================================
# 4. ALGORITMA NAIVE BAYES MURNI
# ==========================================
def pisahkan_berdasarkan_kelas(data):
    # Mengelompokkan data berdasarkan tingkat kemiskinan (kolom terakhir)
    data_terpisah = {}
    for i in range(len(data)):
        baris = data[i]
        kelas = baris[-1]
        if kelas not in data_terpisah:
            data_terpisah[kelas] = []
        # Masukkan semua fitur kecuali kolom target
        data_terpisah[kelas].append(baris[:-1])
    return data_terpisah

def ringkasan_dataset(data):
    # Menghitung rata-rata, standar deviasi, dan jumlah baris untuk setiap kolom
    ringkasan = [(hitung_rata_rata(kolom), hitung_standar_deviasi(kolom), len(kolom)) for kolom in zip(*data)]
    return ringkasan

def latih_naive_bayes(data):
    data_terpisah = pisahkan_berdasarkan_kelas(data)
    ringkasan_model = {}
    for kelas, baris_data in data_terpisah.items():
        ringkasan_model[kelas] = ringkasan_dataset(baris_data)
    return ringkasan_model

def hitung_probabilitas_gaussian(x, rata_rata, stdev):
    # Rumus Probabilitas Gaussian (Kurva Distribusi Normal)
    eksponen = math.exp(-((x - rata_rata)**2 / (2 * stdev**2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * eksponen

def prediksi_satu_baris(ringkasan_model, baris):
    probabilitas = {}
    total_data_latih = sum([ringkasan_model[k][0][2] for k in ringkasan_model])

    for kelas, ringkasan_kelas in ringkasan_model.items():
        # Probabilitas awal (Prior)
        probabilitas[kelas] = math.log(ringkasan_kelas[0][2] / total_data_latih)

        # Menghitung probabilitas setiap fitur dan menjumlahkannya (menggunakan Log)
        for i in range(len(ringkasan_kelas)):
            rata_rata, stdev, _ = ringkasan_kelas[i]
            nilai_x = baris[i]
            prob = hitung_probabilitas_gaussian(nilai_x, rata_rata, stdev)
            # Ditambah 1e-9 untuk menghindari error log(0) jika nilai probabilitas sangat kecil
            probabilitas[kelas] += math.log(prob + 1e-9)

    # Mengembalikan kelas yang memiliki nilai probabilitas paling tinggi
    return max(probabilitas, key=probabilitas.get)

# ==========================================
# 5. EKSEKUSI & EVALUASI
# ==========================================
print("Memulai proses pelatihan model (Training)...")
model_nb = latih_naive_bayes(data_latih)

print("Melakukan prediksi pada data uji (Testing)...")
prediksi_benar = 0
total_uji = len(data_uji)

for baris in data_uji:
    kelas_asli = baris[-1]
    fitur_uji = baris[:-1]
    kelas_tebakan = prediksi_satu_baris(model_nb, fitur_uji)

    if kelas_tebakan == kelas_asli:
        prediksi_benar += 1

akurasi = (prediksi_benar / float(total_uji)) * 100
print(f"Selesai dengan akurasi {akurasi:.2f}%")