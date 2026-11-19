# Klasifikasi Jenis Sampah Rumah Tangga pada Citra Dunia Nyata Menggunakan CNN

## Progres 3 — Perancangan dan implementasi awal model

Judul lengkap tetap **Klasifikasi Jenis Sampah Rumah Tangga pada Citra Dunia Nyata Menggunakan CNN:
Perbandingan Baseline, Augmentasi, dan Transfer Learning**. Progres 3 menambahkan implementasi
PyTorch, diagram, notebook yang telah dieksekusi, pengujian sintetis CPU, serta BAB III.
**Belum ada training, tuning, atau evaluasi dataset. Pelatihan dilanjutkan pada Progres 4.**
Bagian Progres 1–2 di bawah dipertahankan sebagai catatan historis; jadwal tahap selanjutnya
di sana telah digantikan oleh cakupan Progres 3 ini. Augmentasi dan pembobotan kelas merupakan
varian eksperimen terkontrol, bukan pengaturan wajib di semua eksperimen.

| Model | Struktur | Parameter total / trainable pada verifikasi |
| --- | --- | --- |
| CNN baseline | Conv–ReLU–MaxPool ×3 (32/64/128), adaptive average pooling, Linear 128→64, ReLU, dropout 0,3, Linear 64→9 | 102.089 / 102.089 |
| MobileNetV3-Small | torchvision features, average pooling, Linear 576→1024, Hardswish, dropout 0,2, Linear 1024→9 | 1.527.081 / 600.073 |

Keduanya menerima `[B,3,128,128]` dan menghasilkan **logits** `[B,9]`. Gunakan
`nn.CrossEntropyLoss` dengan target integer `torch.long` 0–8; softmax hanya untuk probabilitas
inferensi. Ini padanan objective sparse categorical crossentropy yang disebut pada BAB II.
CNN merencanakan skala piksel `[0,1]`; MobileNet memakai skala `[0,1]` diikuti mean/std ImageNet.
Resize 128 adalah adaptasi proposal terhadap transform bobot standar crop 224, belum dievaluasi.

MobileNet pada verifikasi memakai **bobot acak, tanpa unduhan**, dengan `features` dibekukan
(termasuk statistik BatchNorm) dan seluruh classifier trainable. IMAGENET1K_V1 direncanakan
untuk Progres 4; **bobot pretrained belum dimuat**. API `set_trainable_feature_blocks(N)`
menyiapkan fine-tuning N blok terakhir; buat ulang optimizer setelah mengubah kebijakan.

### Menjalankan bukti Progres 3

Python yang diuji: **3.12.14**, torch **2.7.1+cpu**, torchvision **0.22.1+cpu**. Dependensi model
terpisah dari `requirements.txt` EDA; versi aktual ada di
`results/model_design/dependency_versions.json` dan `requirements-model-lock.txt`.
Tidak ada perubahan driver/CUDA.
Contoh berikut untuk PowerShell; `python` harus menunjuk interpreter Python 3.12 yang tersedia:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-model.txt

# Inisialisasi kedua arsitektur tanpa bobot ImageNet
python -c "from src.models import CNNBaseline, MobileNetV3Small; print(CNNBaseline()); print(MobileNetV3Small(pretrained=False))"

# Forward sintetis, jumlah parameter, finite values, dan freeze policy
python -m src.verify_models
python -m pytest -q --junitxml=results/model_design/pytest_results.xml

# Bangun diagram dan eksekusi notebook dengan kernel interpreter aktif
python scripts/build_model_evidence.py
# Eksekusi ulang notebook saja
python scripts/build_model_evidence.py --notebook-only
```

Perintah verifikasi juga dapat dipanggil sebagai `python src/verify_models.py`. Tidak ada data
mentah atau bobot ImageNet yang dibutuhkan. Kedua output aktual `[2,9]`, finite, dan berulang
identik dalam eval mode. **8 pengujian lulus**; detail di XML hasil tes. Pengujian tidak melakukan
backward atau optimizer step.

| Lokasi | Isi |
| --- | --- |
| `src/models/config.py` | Urutan kelas asli dan kebijakan input |
| `src/models/cnn_baseline.py`, `src/models/mobilenet_v3.py` | Implementasi reusable |
| `configs/model_initial.yaml` | Usulan loss, optimizer, learning rate, batch size, epoch dan eksperimen |
| `src/verify_models.py` | Verifikasi CPU dan factory loss/optimizer tanpa training |
| `results/model_design/` | JSON bukti verifikasi, versi, cuplikan kode, preservasi PDF, dan hasil tes |
| `figures/cnn_baseline_architecture.png`, `figures/mobilenetv3_small_architecture.png` | Diagram aktual; sumber SVG dengan nama sama |
| `notebooks/02_model_design.ipynb` | Notebook dengan output eksekusi asli |
| `laporan/BAB_III_Progres3.docx` | BAB III yang dapat diedit |
| `laporan/DL_A_KelompokART_Progres3.pdf` | PDF lengkap: 10 halaman asli diikuti BAB III |
| `docs/progress3_implementation_report.md` | Temuan inspeksi, hasil, reproduksi, dan keterbatasan |

Usulan CNN: Adam lr 0,001, batch 32, maksimum 30 epoch, seed 42. Usulan MobileNet:
AdamW lr 0,0001, weight decay 0,01, batch 16, maksimum 30 epoch, seed 42, features frozen.
Nilai ini belum dioptimalkan. Pengaturan monitoring hanya rencana, tidak memuat hasil model.

Untuk membangun ulang laporan, pasang `requirements-report.txt`, jalankan
`python scripts/build_progress3_report.py`, lalu `python scripts/export_progress3.py`
dengan LibreOffice tersedia, dan `python scripts/validate_progress3.py`. LibreOffice lokal sesi ini
tersedia di cache proyek (tidak masuk Git). Alternatif ekspor DOCX ke PDF dapat dilakukan manual;
simpan sebagai `laporan/BAB_III_Progres3.pdf` sebelum validasi.
Validator menggabungkan PDF, membandingkan raster dan teks seluruh 10 halaman lama, dan
memeriksa pagination serta bukti notebook. Halaman BAB I–II tidak ditulis ulang.

Sumber teknis: [MobileNetV3-Small torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html),
[CrossEntropyLoss PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

---

Repositori bukti pengerjaan **Progres 1–2** mata kuliah Deep Learning: eksplorasi data (EDA),
audit kualitas dataset, dan rencana *preprocessing* serta pembagian data untuk klasifikasi
9 kelas material sampah pada dataset **RealWaste**.

Hasil pelatihan model **belum** dilaporkan pada tahap ini sesuai ketentuan Progres 2.

## Identitas

| Komponen | Keterangan |
| --- | --- |
| Mata kuliah | Deep Learning (Kelas A) |
| Program studi | Informatika, Fakultas Teknik, Universitas Siliwangi |
| Kelompok | Kelompok ART (Aar, Rizki, Try) |
| Anggota | Try Apriyana Nugraha (237006067), Aar Lazuardi Majid (237006070), Muhammad Rizki (237006085) |
| Jenis tugas | Klasifikasi citra multi-kelas |
| Laporan | `laporan/DL_A_KelompokART_Progres2.pdf` |

## Dataset

| Komponen | Keterangan |
| --- | --- |
| Nama | RealWaste |
| Sumber | UCI Machine Learning Repository, id 908 — https://archive.ics.uci.edu/dataset/908/realwaste |
| Alternatif publik | Kaggle — https://www.kaggle.com/datasets/joebeachcapital/realwaste (tautan dataset pada laporan §2.9) |
| DOI | 10.24432/C5SS4G |
| Lisensi | CC BY-NC-SA 4.0 (README dataset); halaman UCI mencantumkan CC BY 4.0. Penggunaan hanya untuk penelitian/pendidikan non-komersial |
| Ukuran arsip | 688,5 MB (`realwaste.zip`) |
| SHA-256 arsip | `1ede08b32358ee62065bcc1c8cb47a2ece04e8dff5b1fb53342bb8750895b2f3` |
| Jumlah | 4.752 citra JPG, 524 × 524 piksel, RGB, 9 kelas |

Data mentah **tidak** disimpan di repositori. Pengunduhan arsip resmi UCI dilakukan otomatis oleh
notebook (`notebooks/01_eda_realwaste.ipynb`); server UCI tidak mendukung *resume*, jadi unduhan
diulang dari awal bila terputus. Untuk pemakaian lokal, arsip diunduh manual (lihat bagian
*Cara menjalankan*).

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
├── laporan/                    # PDF laporan Progres 2 (DL_A_KelompokART_Progres2.pdf)
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

## Referensi

Seluruh DOI di bawah telah diperiksa melalui API Crossref/DataCite dan diarahkan ke halaman
penerbit (14 November 2026); tidak ada referensi hasil rekaan.

1. S. Single, S. Iranmanesh, and R. Raad, “RealWaste: A novel real-life data set for landfill
   waste classification using deep learning,” *Information*, vol. 14, no. 12, art. 633, 2023,
   doi: 10.3390/info14120633.
2. S. Single, S. Iranmanesh, and R. Raad, *RealWaste*, UCI Machine Learning Repository, 2023,
   doi: 10.24432/C5SS4G.
3. N. Nnamoko, J. Barrowclough, and J. Procter, “Solid waste image classification using deep
   convolutional neural network,” *Infrastructures*, vol. 7, no. 4, art. 47, 2022,
   doi: 10.3390/infrastructures7040047.
4. D. Verber, T. Grneva, and J. Dugonik, “Image-based waste classification using a hybrid deep
   learning architecture with transfer learning and edge AI,” *Mathematics*, vol. 14, no. 7,
   art. 1176, 2026, doi: 10.3390/math14071176.
5. R. A. Kumala, C. A. Sari, and E. H. Rachmawanto, “A comparison of MobileNetV2 and VGG16
   architectures with transfer learning for multi-class image-based waste classification,”
   *Journal of Applied Informatics and Computing*, vol. 9, no. 4, pp. 1610–1624, 2025,
   doi: 10.30871/jaic.v9i4.9958.
6. M. A. Saputra, “Class imbalance in deep learning-based image classification: A systematic
   review,” *Journal of Applied Informatics and Computing*, vol. 10, no. 5, pp. 4978–4989, 2026,
   doi: 10.30871/jaic.v10i5.13881.
7. C. J. Hellín, A. A. Olmedo, A. Valledor, J. Gómez, M. López-Benítez, and A. Tayebi,
   “Unraveling the impact of class imbalance on deep-learning models for medical image
   classification,” *Applied Sciences*, vol. 14, no. 8, art. 3419, 2024,
   doi: 10.3390/app14083419.
8. C. Shorten and T. M. Khoshgoftaar, “A survey on image data augmentation for deep learning,”
   *Journal of Big Data*, vol. 6, no. 1, art. 60, 2019, doi: 10.1186/s40537-019-0197-0.
9. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual
   explanations from deep networks via gradient-based localization,” in *Proc. IEEE Int. Conf.
   Computer Vision (ICCV)*, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.
10. A. Howard et al., “Searching for MobileNetV3,” in *Proc. IEEE/CVF Int. Conf. Computer Vision
    (ICCV)*, 2019, pp. 1314–1324, doi: 10.1109/ICCV.2019.00140.
