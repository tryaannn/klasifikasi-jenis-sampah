# Laporan implementasi Progres 3 RealWaste

Tanggal: 18 November 2026 (Asia/Jakarta). Kelompok ART, Deep Learning Kelas A.

Implementasi berhenti pada desain arsitektur, konfigurasi awal, pengujian sintetis, notebook,
diagram, dan laporan BAB III. Tidak ada training, tuning, backward pass, optimizer step,
evaluasi dataset, aplikasi web, deployment, commit, atau push.

## Temuan inspeksi awal

- Working tree awal bersih. Tidak ditemukan AGENTS.md atau kode model sebelumnya.
- Struktur awal: `src/eda_realwaste.py`, notebook EDA, requirements EDA, `data/README.md`,
  empat gambar EDA, enam keluaran CSV/JSON, README, serta PDF Progres 2 sebanyak 10 halaman.
- README, seluruh source EDA, struktur/sel notebook 01, requirements, hasil EDA, dan seluruh
  teks PDF dibaca. Tata letak referensi diperiksa: A4, Aptos 10,5 pt, judul Calibri, footer halaman.
- Laporan mencatat 4.752 citra RGB 524 × 524, sembilan label, input rencana 128 × 128,
  CNN tiga blok dan MobileNetV3-Small. Audit korupsi/duplikat identik selesai; kandidat
  near-duplicate memerlukan tinjauan manual lanjutan. Raw images tidak disertakan dan tidak diunduh.
- Split, cache resize, augmentasi, dan pipeline training belum dilaksanakan. Ini batas cakupan,
  bukan kegagalan verifikasi model. Rencana 70/15/15 tetap stratified dan group-aware; jumlah
  3.326/713/713 belum menjadi jumlah split aktual.
- Python/py tidak tersedia di PATH. Runtime terbundel ditemukan melalui manifest
  `codex-primary-runtime/runtime.json`: Python 3.12.14. Dibuat `.venv` proyek terisolasi;
  dependensi dipasang di sana, tanpa mengubah interpreter asal atau GPU/CUDA.

## Berkas yang dibuat dan diubah

Dibuat: `AGENTS.md`, `requirements-model.txt`, `requirements-model-lock.txt`, `requirements-report.txt`,
`configs/model_initial.yaml`, empat berkas `src/models/`, `src/verify_models.py`,
`tests/test_models.py`, `notebooks/02_model_design.ipynb`, dua diagram PNG dan SVG,
`scripts/build_model_evidence.py`, `scripts/build_progress3_report.py`,
`scripts/export_progress3.py`, `scripts/validate_progress3.py`, dokumen laporan BAB III,
PDF gabungan, serta artefak JSON/XML di `results/model_design/`.
README ditambah panduan Progres 3; `.gitignore` ditambah pengecualian cache dan QA.
Source, notebook, hasil dan gambar EDA serta PDF asli tidak diubah.

## Implementasi arsitektur

CNN: Conv2d 3→32→64→128, masing-masing kernel 3, padding 1, ReLU, MaxPool2d(2).
Peta fitur: 32×64×64 → 64×32×32 → 128×16×16. AdaptiveAvgPool2d(1), flatten,
Linear 128→64, ReLU, Dropout 0,3, Linear 64→9. Dibangun dari awal.

MobileNet: implementasi resmi torchvision MobileNetV3-Small, features menghasilkan
576×4×4 pada input 128. Average pooling, flatten, Linear 576→1024, Hardswish,
Dropout 0,2, Linear pengganti 1024→9. Seluruh features frozen dan classifier trainable.
Frozen BatchNorm tetap eval meskipun wrapper menerima `.train()`; buffer statistiknya diuji
tidak berubah. `set_trainable_feature_blocks(N)` memungkinkan fine-tuning selektif kemudian;
optimizer perlu dibuat ulang setelah pergantian policy. Seed constructor dipertahankan dalam
scope RNG lokal untuk inisialisasi yang dapat diulang.

Kedua model menghasilkan logits, tanpa softmax internal. Target: `torch.long` `[B]`, ID 0–8
berurutan Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic,
Textile Trash, Vegetation. CrossEntropyLoss adalah padanan objective sparse categorical
crossentropy pada BAB II, dengan logits langsung. Softmax dipakai di luar model untuk probabilitas.

Kebijakan input rencana terpisah: CNN skala [0,1]; MobileNet skala [0,1] kemudian normalisasi
mean [0.485,0.456,0.406], std [0.229,0.224,0.225]. Resize 128 sesuai proposal berbeda dari
crop 224 pada transform bobot resmi. Pipeline citra belum dibangun; tensor acak hanya menguji struktur.

## Bukti eksekusi aktual

| Model | Input → output | Total parameter | Trainable | Finite | Eval deterministik |
| --- | --- | ---: | ---: | --- | --- |
| CNN baseline | [2,3,128,128] → [2,9] | 102.089 | 102.089 | Ya | Ya |
| MobileNetV3-Small | [2,3,128,128] → [2,9] | 1.527.081 | 600.073 | Ya | Ya |

Eksekusi CPU dengan dua thread, seed 42, eval mode dan inference mode. Backbone frozen
terverifikasi; parameter features 927.008. Tidak ada GPU yang digunakan. ImageNet **belum
diunduh/dimuat**; verifikasi menggunakan `pretrained=False`. Konfigurasi pelatihan berikutnya
merencanakan IMAGENET1K_V1. Membekukan backbone acak di sini hanya pengujian API.

Artefak: `model_verification.json`, `baseline_architecture_summary.json`, dan
`mobilenet_architecture_summary.json`. Nilai parameter dihitung dari objek model aktual,
bukan dari tabel model ImageNet 1.000 kelas. Timestamp disimpan dalam UTC.

`pytest`: **8 passed** (eksekusi awal 4,37 detik; hasil terakhir tercatat dalam XML).
Masing-masing kasus mencakup beberapa pemeriksaan terkait: shape, finite, kelas/output,
class mapping terhadap CSV historis, parameter count, freeze/BatchNorm buffers, fine-tuning
selektif dan batas N, seed, konfigurasi, loss dengan integer labels, konversi softmax,
determinisme eval, optimizer hanya parameter trainable, serta integritas JSON.
Tidak ada pengunduhan dalam ordinary tests.

Notebook memiliki tujuh sel kode yang dieksekusi dengan kernel Python aktif, tanpa error.
Model diimpor dari `src/models/`; output asli mencakup arsitektur, shape, parameter counts,
status bobot, konfigurasi optimizer, dan diagram. Tidak ada metrik training/evaluasi dataset.

Diagram dibuat secara reproducible dengan Matplotlib; dimensi stage diambil dari forward hooks
implementasi aktual. SVG adalah sumber editable; generator ada di `scripts/build_model_evidence.py`.
Softmax ditandai sebagai konversi inferensi di luar model. Tidak memakai gambar arsitektur eksternal.

## Konfigurasi awal

| Komponen | CNN | MobileNet |
| --- | --- | --- |
| Loss | CrossEntropyLoss | CrossEntropyLoss |
| Optimizer | Adam | AdamW |
| Learning rate | 0.001 | 0.0001 |
| Weight decay | 0.0 | 0.01 |
| Batch size rencana | 32 | 16 |
| Maksimum epoch rencana | 30 | 30 |
| Seed | 42 | 42 |
| Bobot rencana | Acak | IMAGENET1K_V1, features frozen |

Adam menjadi titik awal baseline; learning rate lebih kecil untuk adaptasi transfer learning.
Batch MobileNet lebih kecil untuk membatasi memori. Global pooling dan dropout membatasi
kapasitas baseline. Semua pilihan merupakan proposal; kelayakan training pada CPU 8 GB
belum diuji. Augmentasi dan class weighting dinonaktifkan pada kontrol dan akan dibandingkan
sebagai varian terpisah. Loss/accuracy/precision/recall/F1 makro adalah rencana monitoring,
bukan hasil. Confusion matrix, gap latih–validasi, waktu training, dan Grad-CAM tetap direncanakan.

## Lingkungan dan reproduksi

Versi aktual lengkap disimpan di `results/model_design/dependency_versions.json`.
Python 3.12.14, torch 2.7.1+cpu, torchvision 0.22.1+cpu. Dependensi EDA asli tidak diubah.
Pasangan torch/torchvision CPU dipatok eksplisit; versi dependensi lain dicatat dari lingkungan aktual.

```powershell
# Dengan Python 3.12 tersedia di PATH
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-model.txt
python -m src.verify_models
python -m pytest -q --junitxml=results/model_design/pytest_results.xml
python scripts/build_model_evidence.py
python -m pip install -r requirements-report.txt
python scripts/build_progress3_report.py
python scripts/export_progress3.py
python scripts/validate_progress3.py
```

Pada mesin sesi ini, perintah Python menggunakan `.venv/Scripts/python.exe`, dengan basis
runtime terbundel. Instalasi membutuhkan akses jaringan untuk paket; verifikasi/tests/notebook
setelah instalasi tidak memerlukan jaringan, dataset atau weights. Notebook di browser juga
dapat dieksekusi dengan kernel environment yang sama. Generator notebook tidak melatih model.

## Laporan dan preservasi historis

`laporan/BAB_III_Progres3.docx` hanya memuat BAB III yang dapat diedit. Ekspor standalone:
`laporan/BAB_III_Progres3.pdf`. Dokumen lengkap: `laporan/DL_A_KelompokART_Progres3.pdf`.
Original BAB I–II beserta identitas, daftar referensi dan seluruh gambar tetap 10 halaman pertama.
Tidak dibuat unified editable DOCX karena akan berisiko merusak fidelity PDF historis.

`render_docx.py` awalnya gagal karena LibreOffice `soffice.exe` tidak tersedia di PATH.
Ekspor Microsoft Word 16.0 terpasang terhenti; helper ekspor dihentikan dan hanya dokumen
BAB III yang ditutup untuk melepas file lock. Dokumen pengguna lainnya tidak diedit.
LibreOffice 26.8.3 diunduh dari situs resmi dan diekstrak sebagai administrative image ke cache
proyek, tanpa instalasi global. Renderer kemudian dijalankan dengan LibreOffice lokal dan
Poppler terbundel; rasterisasi PyMuPDF dipakai pula untuk pemeriksaan preservasi PDF.
Tidak menggambar ulang teks DOCX menjadi PDF melalui generator lain.
BAB III menggunakan Calibri 10,5 pt karena Aptos yang tertanam pada PDF historis tidak
tersedia sebagai font lokal renderer. Font dan tampilan halaman BAB I–II tetap identik.
Pagination BAB III dimulai halaman 11. Validator membandingkan raster, teks dan ukuran
semua halaman lama, SHA-256 original terhadap Git HEAD, serta seluruh bukti EDA historis.
Hasil terukur disimpan dalam `results/model_design/report_preservation.json`.
Cuplikan kode diekstrak langsung dari source dan divalidasi dalam `report_code_excerpts.json`.

Hasil akhir: PDF gabungan **17 halaman** (10 halaman asli + 7 halaman BAB III bernomor 11–17).
Seluruh 10 halaman historis identik secara raster dan teks, serta PDF asli identik byte demi byte
dengan Git HEAD (SHA-256 `6ad7a31b1618457d408db36127735997d076204d203f618bc4fe1c17c61dc51b`).
Seluruh berkas bukti historis selain README dan `.gitignore` sesuai Git HEAD dengan filter
checkout CRLF yang sudah berlaku. Semua halaman BAB III ditinjau visual; ekspor yang dikirim
identik dengan hasil render DOCX yang diperiksa. Pagination, batas halaman, tujuh sel kode
notebook tanpa error, dan kecocokan cuplikan sumber lolos pemeriksaan otomatis.

BAB III hanya subseksi 3.1–3.5. Isi mencakup tabel kesiapan data, alasan MobileNet sebagai
transfer learning utama dan CNN pembanding, spesifikasi/mapping kelas, diagram kedua model,
konfigurasi dari YAML, potongan kode, status bobot dan batasan tanpa klaim performa.
Jadwal BAB II yang menyebut training pada Progres 3 dijelaskan sebagai rencana historis;
cakupan terbaru memindahkan training ke Progres 4. Terminologi loss diperjelas tanpa mengedit BAB II.

## Kesiapan Progres 4 dan batasan

Arsitektur, API, konfigurasi dan bukti pengujian siap sebagai dasar Progres 4. Sebelum training,
tuntaskan review near-duplicate/label ambigu, tetapkan grup serta split yang tidak bocor,
bangun transform dan loader, lalu muat bobot pretrained secara eksplisit dan catat keberhasilannya.
Gunakan split yang sama untuk semua eksperimen; hitung class weights dari training saja,
augmentasi training saja, pilih konfigurasi lewat validation, dan evaluasi testing satu kali di akhir.
Tidak ada klaim keberhasilan transfer learning atau perbandingan performa sebelum tahap itu.

## Sumber teknis

- [UCI RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste), identitas dan spesifikasi dataset.
- [Torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html), constructor, weights dan normalisasi.
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), logits dan integer targets.
- [Howard et al., Searching for MobileNetV3](https://arxiv.org/abs/1905.02244), dasar arsitektur; sejalan referensi [10] pada BAB II.

Sumber resmi diakses 18 November 2026. Fakta historis EDA berasal dari laporan dan artefak
repositori yang dipertahankan, tanpa mengulang pemrosesan raw dataset.
