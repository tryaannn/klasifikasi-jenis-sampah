# Klasifikasi Jenis Sampah Rumah Tangga pada Citra Dunia Nyata Menggunakan CNN

Repositori bukti pengerjaan **Progres 1–2** mata kuliah Deep Learning: eksplorasi data (EDA),
audit kualitas dataset, dan rencana *preprocessing* serta pembagian data untuk klasifikasi
9 kelas material sampah pada dataset **RealWaste**.

Hasil pelatihan model **belum** dilaporkan pada tahap ini sesuai ketentuan Progres 2.

## Identitas

| Komponen | Keterangan |
| --- | --- |
| Mata kuliah | Deep Learning (Kelas A) |
| Program studi | Informatika, Fakultas Teknik, Universitas Siliwangi |
| Kelompok | Kelompok 3 |
| Anggota | Try Apriyana Nugraha (237006067), Aar Lazuardi Majid (237006070), Muhammad Rizki (237006085) |
| Jenis tugas | Klasifikasi citra multi-kelas |
| Laporan | `laporan/DL_A_Kelompok3_Progres2.pdf` |

## Dataset

| Komponen | Keterangan |
| --- | --- |
| Nama | RealWaste |
| Sumber | UCI Machine Learning Repository, id 908 — https://archive.ics.uci.edu/dataset/908/realwaste |
| Cermin | Kaggle — https://www.kaggle.com/datasets/joebeachcapital/realwaste |
| DOI | 10.24432/C5SS4G |
| Lisensi | CC BY-NC-SA 4.0 (README dataset); halaman UCI mencantumkan CC BY 4.0. Penggunaan hanya untuk penelitian/pendidikan non-komersial |
| Ukuran arsip | 688,5 MB (`realwaste.zip`) |
| SHA-256 arsip | `1ede08b32358ee62065bcc1c8cb47a2ece04e8dff5b1fb53342bb8750895b2f3` |
| Jumlah | 4.752 citra JPG, 524 × 524 piksel, RGB, 9 kelas |

Data mentah **tidak** disimpan di repositori. Skrip EDA mengunduh arsip resmi UCI secara otomatis
(resume tidak didukung server, jadi unduhan diulang dari awal bila terputus).

## Hasil pemeriksaan dataset (ringkas)

Distribusi kelas (total 4.752 citra, rasio kelas terbesar : terkecil = 2,90 : 1):

| Kelas | Jumlah | Persentase |
| --- | --- | --- |
| Plastic | 921 | 19,38 % |
| Metal | 790 | 16,62 % |
| Paper | 500 | 10,52 % |
| Miscellaneous Trash | 495 | 10,42 % |
| Cardboard | 461 | 9,70 % |
| Vegetation | 436 | 9,18 % |
| Glass | 420 | 8,84 % |
| Food Organics | 411 | 8,65 % |
| Textile Trash | 318 | 6,69 % |

Temuan lain:

- Tidak ada berkas rusak, berkas 0 byte, atau duplikat identik (MD5).
- Kemiripan tinggi (*difference hash* 64 bit): 25 pasangan berjarak ≤ 4 (17 klaster, 41 citra,
  0,86 %), termasuk 8 pasangan lintas kelas; pada ambang ≤ 6 ditemukan 213 pasangan (216 citra)
  dengan satu klaster berisi 84 citra dari berbagai kelas karena latar identik.
- Kecerahan rata-rata 157,14 (median 157,98; rentang 107,48–184,92); hanya 18 citra < 120.
- Kontras median antarkelas 32,8 (Glass) sampai 56,0 (Vegetation).
- Ukuran berkas 48–243 KB (median 134 KB); seluruh citra beresolusi sama.

Implikasi metodologis: pembagian data harus **terstratifikasi** dan **group-aware** (citra dalam
satu klaster kemiripan tidak boleh melintasi subset), evaluasi memakai **F1 makro**, dan pelatihan
memakai augmentasi serta pembobotan kelas.

## Struktur repositori

```
.
├── src/eda_realwaste.py        # skrip EDA (inventaris, statistik, duplikat, gambar)
├── notebooks/                  # notebook Colab yang menjalankan skrip di atas
├── results/                    # keluaran EDA: CSV, JSON (bukti mentah)
├── figures/                    # gambar EDA yang dipakai di laporan
├── laporan/                    # PDF laporan Progres 2
├── data/                       # (kosong) tempat data mentah bila dijalankan lokal
├── requirements.txt
└── README.md
```

## Cara menjalankan

### Lokal

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# unduh arsip dari https://archive.ics.uci.edu/static/public/908/realwaste.zip
# lalu ekstrak ke data/ sehingga terbentuk data/realwaste-main/RealWaste/<kelas>/*.jpg

python src/eda_realwaste.py --raw data/realwaste-main/RealWaste --out results
```

### Google Colab

Buka `notebooks/01_eda_realwaste.ipynb`, jalankan seluruh sel. Notebook mengunduh dataset,
menjalankan skrip EDA, lalu menampilkan tabel dan gambar hasilnya.

Argumen skrip:

- `--raw` folder induk yang berisi subfolder kelas (default `dataset/realwaste-main/RealWaste`)
- `--out` folder keluaran CSV/JSON/gambar (default `analysis`)

## Keluaran skrip

| Berkas | Isi |
| --- | --- |
| `results/inventory.csv` | satu baris per citra: kelas, resolusi, ukuran, kecerahan, kontras, MD5, dHash |
| `results/class_distribution.csv` | jumlah dan persentase per kelas, rasio terhadap kelas terkecil |
| `results/near_duplicates.csv` | pasangan citra dengan jarak dHash ≤ 6 beserta penanda beda kelas |
| `results/resolutions.csv` | distribusi resolusi |
| `results/summary.json` | ringkasan pemeriksaan (jumlah, duplikat, kecerahan, ukuran) |
| `results/sample_per_kelas.json` | contoh citra per kelas (seed 42) |
| `figures/eda_01..04.png` | distribusi kelas, kecerahan/ukuran, kecerahan per kelas, contoh per kelas |

## Rencana tahap berikutnya (Progres 3)

1. *Preprocessing*: resize 128 × 128, cache uint8 (≈ 233 MB), normalisasi sesuai arsitektur,
   augmentasi hanya pada data training.
2. Model: (a) CNN baseline tiga blok konvolusi; (b) MobileNetV3-Small *transfer learning*.
3. Pembagian data: 70 % training (3.326), 15 % validasi (713), 15 % pengujian (713),
   terstratifikasi dan group-aware, seed 42, pengujian dievaluasi satu kali.
4. Evaluasi: akurasi, precision/recall/F1 makro, *confusion matrix*, selisih akurasi
   latih–validasi, jumlah parameter, waktu pelatihan, dan Grad-CAM.

Target perangkat: CPU dengan memori 8 GB (tanpa GPU).

## Referensi utama

1. S. Single, S. Iranmanesh, and R. Raad, “RealWaste: A novel real-life data set for landfill
   waste classification using deep learning,” *Information*, vol. 14, no. 12, art. 633, 2023,
   doi: 10.3390/info14120633.
2. S. Single, S. Iranmanesh, and R. Raad, *RealWaste*, UCI Machine Learning Repository, 2023,
   doi: 10.24432/C5SS4G.
3. N. Nnamoko, J. Barrowclough, and J. Procter, “Solid waste image classification using deep
   convolutional neural network,” *Infrastructures*, vol. 7, no. 4, art. 47, 2022,
   doi: 10.3390/infrastructures7040047.
4. C. J. Hellín et al., “Unraveling the impact of class imbalance on deep-learning models for
   medical image classification,” *Applied Sciences*, vol. 14, no. 8, art. 3419, 2024,
   doi: 10.3390/app14083419.
5. A. Howard et al., “Searching for MobileNetV3,” in *Proc. IEEE/CVF ICCV*, 2019, pp. 1314–1324,
   doi: 10.1109/ICCV.2019.00140.
