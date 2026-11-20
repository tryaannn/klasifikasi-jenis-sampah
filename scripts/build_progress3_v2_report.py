"""Build polished, publication-ready Progres 3 v2 Report (DOCX).

Combines historical BAB I and BAB II from Progres 2 with the newly enriched
BAB III (v2), featuring actual group-aware data split metrics (3,327/713/712),
computational complexity profiling (FLOPs/MACs/RAM feasibility),
and anti-leakage class weighting.

Key refinements in v2:
1. Unified Footers across all sections:
   - Section 0 footer updated from 'Progres 1–2' to 'Progres 3'.
   - Section 1 linked to Section 0 (is_linked_to_previous = True).
   - Removed pgNumType from Section 1 so page numbers naturally continue from 10 to 11..16.
   - Removed empty trailing paragraph in footer.
2. Academic Table & Figure Formatting:
   - Formal Table Captions placed ABOVE each table (Tabel 3.1 .. Tabel 3.8).
   - Formal Figure Captions placed BELOW each figure (Gambar 3.1 .. Gambar 3.4).
   - Added Gambar 3.1 (data_split_distribution.png) and Gambar 3.4 (data_augmentation_preview.png).
   - Cell padding (top/bottom 90 dxa, left/right 120 dxa), light borders (#D3D3D3), and header shading (#EBF1F5).
3. Typography & Spacing:
   - Calibri 10.5pt, 1.15 line spacing, 6pt space after for body text.
   - Headings styled cleanly with Calibri bold in true black.
   - Code blocks with Consolas 8pt, bordered container, and keep_together=True.
4. Preserved 100% of BAB I and BAB II from base document DL_A_KelompokART_Progres2.docx.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
SCRATCH = Path(r"C:\Users\asus\.gemini\antigravity-cli\brain\d0e1b361-44e2-45d7-b366-b9c7e5cc5bdb\scratch")


def format_table(table, col_widths, bg_header="EBF1F5"):
    """Format table with elegant academic styling, padding, and borders."""
    for ri, row in enumerate(table.rows):
        pr = row._tr.get_or_add_trPr()
        pr.append(OxmlElement("w:cantSplit"))
        if ri == 0:
            pr.append(OxmlElement("w:tblHeader"))
        for ci, cell in enumerate(row.cells):
            cell.width = Inches(col_widths[ci])
            cell.vertical_alignment = 1  # Center vertical
            tcpr = cell._tc.get_or_add_tcPr()

            # Borders
            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right"):
                e = OxmlElement(f"w:{edge}")
                for key, val in (("val", "single"), ("sz", "4"), ("color", "D3D3D3")):
                    e.set(qn(f"w:{key}"), val)
                borders.append(e)
            tcpr.append(borders)

            # Shading
            shading = OxmlElement("w:shd")
            shading.set(qn("w:fill"), bg_header if ri == 0 else "FFFFFF")
            tcpr.append(shading)

            # Padding (margins): top=90, bottom=90, left=120, right=120 dxa
            margins = OxmlElement("w:tcMar")
            for edge, val in (("top", "90"), ("bottom", "90"), ("left", "120"), ("right", "120")):
                e = OxmlElement(f"w:{edge}")
                e.set(qn("w:w"), val)
                e.set(qn("w:type"), "dxa")
                margins.append(e)
            tcpr.append(margins)

            for para in cell.paragraphs:
                para.paragraph_format.space_after = Pt(2)
                para.paragraph_format.space_before = Pt(2)
                para.paragraph_format.line_spacing = 1.05
                for run in para.runs:
                    run.font.name = "Calibri"
                    run.font.size = Pt(9.0 if len(col_widths) >= 5 else 9.5)
                    run.bold = (ri == 0)


def build_v2_reports():
    print("Loading empirical metadata and profiling results...")
    split_summary = json.loads((ROOT / "results/data_split_summary.json").read_text(encoding="utf-8"))
    weights_data = json.loads((ROOT / "results/model_design/class_weights.json").read_text(encoding="utf-8"))
    profiling_data = json.loads((ROOT / "results/model_design/hardware_profiling.json").read_text(encoding="utf-8"))
    verification_data = json.loads((ROOT / "results/model_design/model_verification.json").read_text(encoding="utf-8"))

    base_docx_path = PARENT / "Laporan-hasil/DL_A_KelompokART_Progres2.docx"
    print(f"Loading base document: {base_docx_path}")
    doc = Document(str(base_docx_path))

    # --- 1. REFINE SECTION 0 & COVER ---
    # Update subtitle on cover
    for p_elem in doc.paragraphs[:6]:
        if "PROGRES 1 DAN PROGRES 2" in p_elem.text:
            p_elem.text = "PROGRES 1, 2, DAN 3 (RINGKAS)  |  PERTEMUAN 10–12"
            p_elem.runs[0].bold = True
            p_elem.runs[0].font.size = Pt(11)
            p_elem.runs[0].font.name = "Calibri"

    # Fix Section 0 Footer
    sec0 = doc.sections[0]
    for para in sec0.footer.paragraphs:
        for r in para.runs:
            if "1–2" in r.text or "1-2" in r.text:
                r.text = r.text.replace("1–2", "3").replace("1-2", "3")
        for r in para.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(8)

    # Remove extra empty paragraph in footer if present
    if len(sec0.footer.paragraphs) > 1:
        for p_extra in sec0.footer.paragraphs[1:]:
            if not p_extra.text.strip():
                p_extra._p.getparent().remove(p_extra._p)

    # --- 2. ADD SECTION 1 (BAB III) WITH LINKED FOOTER ---
    new_section = doc.add_section(WD_SECTION_START.NEW_PAGE)
    new_section.top_margin = Inches(0.85)
    new_section.bottom_margin = Inches(0.85)
    new_section.left_margin = Inches(1.05)
    new_section.right_margin = Inches(0.95)
    new_section.footer_distance = Inches(0.55)

    # Keep footer linked to Section 0 so both sections share the exact same unified footer!
    new_section.footer.is_linked_to_previous = True

    # Remove w:pgNumType from new_section so page numbers continue smoothly 10 -> 11..16
    pg_num_type = new_section._sectPr.find(qn("w:pgNumType"))
    if pg_num_type is not None:
        new_section._sectPr.remove(pg_num_type)

    # Helper functions for layout
    def p(text, *, bold=False, italic=False):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after = Pt(6)
        run = para.add_run(text)
        run.bold = bold
        run.italic = italic
        run.font.name = "Calibri"
        run.font.size = Pt(10.5)
        return para

    def heading(text, level=2):
        para = doc.add_heading(text, level=level)
        para.paragraph_format.space_before = Pt(12)
        para.paragraph_format.space_after = Pt(4)
        para.paragraph_format.keep_with_next = True
        for r in para.runs:
            r.font.name = "Calibri"
            r.font.color.rgb = RGBColor(0, 0, 0)
        return para

    def table_caption(number_str, title_str):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        para.paragraph_format.space_before = Pt(8)
        para.paragraph_format.space_after = Pt(3)
        para.paragraph_format.keep_with_next = True
        r_num = para.add_run(f"{number_str} ")
        r_num.bold = True
        r_num.font.name = "Calibri"
        r_num.font.size = Pt(9.5)
        r_title = para.add_run(title_str)
        r_title.italic = True
        r_title.font.name = "Calibri"
        r_title.font.size = Pt(9.5)
        return para

    def table(headers, rows, widths, number_str="", title_str=""):
        if number_str and title_str:
            table_caption(number_str, title_str)
        tbl = doc.add_table(rows=1, cols=len(headers))
        tbl.autofit = False
        for i, title in enumerate(headers):
            tbl.rows[0].cells[i].text = title
        for row_vals in rows:
            cells = tbl.add_row().cells
            for ci, val in enumerate(row_vals):
                cells[ci].text = str(val)
        format_table(tbl, widths)
        p_space = doc.add_paragraph()
        p_space.paragraph_format.space_before = Pt(0)
        p_space.paragraph_format.space_after = Pt(4)
        return tbl

    def figure(filename, caption_num, caption_text, width_inch=5.8):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after = Pt(2)
        para.paragraph_format.keep_with_next = True
        run = para.add_run()
        run.add_picture(str(ROOT / "figures" / filename), width=Inches(width_inch))
        
        cap_p = doc.add_paragraph()
        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_p.paragraph_format.space_before = Pt(2)
        cap_p.paragraph_format.space_after = Pt(8)
        r_num = cap_p.add_run(f"{caption_num} ")
        r_num.bold = True
        r_num.font.name = "Calibri"
        r_num.font.size = Pt(9.0)
        r_text = cap_p.add_run(caption_text)
        r_text.italic = True
        r_text.font.name = "Calibri"
        r_text.font.size = Pt(9.0)

    def code_block(code_text):
        para = doc.add_paragraph()
        para.paragraph_format.line_spacing = 1.05
        para.paragraph_format.space_before = Pt(3)
        para.paragraph_format.space_after = Pt(6)
        para.paragraph_format.left_indent = Inches(0.2)
        para.paragraph_format.right_indent = Inches(0.2)
        para.paragraph_format.keep_together = True
        
        pPr = para._p.get_or_add_pPr()
        pBdr = parse_xml(
            f'<w:pBdr {nsdecls("w")}>'
            f'  <w:left w:val="single" w:sz="12" w:space="8" w:color="4A7BB0"/>'
            f'  <w:top w:val="none"/>'
            f'  <w:right w:val="none"/>'
            f'  <w:bottom w:val="none"/>'
            f'</w:pBdr>'
        )
        pPr.append(pBdr)
        
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F6F9"/>')
        pPr.append(shd)

        run = para.add_run(code_text)
        run.font.name = "Consolas"
        run.font.size = Pt(8.0)
        run.font.color.rgb = RGBColor(35, 35, 35)

    # --- BAB III CONTENT (v2 REFINED) ---
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(14)
    title_p.paragraph_format.space_after = Pt(8)
    title_p.paragraph_format.keep_with_next = True
    run_t = title_p.add_run("BAB III — PERANCANGAN DAN IMPLEMENTASI MODEL (PROGRES 3)")
    run_t.bold = True
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(14)

    p("Pertemuan 12 | Fokus pada perancangan arsitektur model (3.1–3.4) dan cuplikan baris program implementasi awal (3.5). "
      "Sesuai rubrik evaluasi resmi mata kuliah Deep Learning, proses pelatihan bobot secara menyeluruh dan evaluasi uji "
      "dilaksanakan pada Progres 4 berikutnya. Dokumen versi kedua (v2) ini memperkuat naskah perancangan awal dengan "
      "menyajikan bukti empiris terverifikasi: partisi dataset sadar-kelompok (group-aware partition) dengan jaminan nol kebocoran data, "
      "profiling kompleksitas teoritis hardware (FLOPs, MACs, dan kelayakan alokasi RAM 8 GB CPU), "
      "serta formulasi pembobotan kelas penyeimbang ketimpangan dataset yang diturunkan eksklusif dari data latih.", italic=True)

    # Subbab 3.1
    heading("3.1 Ringkasan Studi Kasus dan Kesiapan Data")
    p("Progres 3 melanjutkan penelitian klasifikasi sembilan kategori material sampah rumah tangga dunia nyata "
      "menggunakan dataset RealWaste (UCI Machine Learning Repository ID: 908). Dataset ini memiliki tantangan riil berupa variasi latar "
      "belakang tidak seragam, pencahayaan bebas, serta ketimpangan frekuensi sampel antarkelas. Seluruh modul perancangan telah "
      "direalisasikan secara modular dan tervalidasi menggunakan pengujian unit otomatis.")

    table(["Komponen Analisis", "Spesifikasi Implementasi Aktual Mahasiswa"], [
        ("Judul Proyek Resmi", "Klasifikasi Jenis Sampah Rumah Tangga pada Citra Dunia Nyata Menggunakan CNN: Perbandingan Baseline, Augmentasi, dan Transfer Learning"),
        ("Tugas dan Target Klasifikasi", "Klasifikasi citra multi-kelas (9 kelas material eksklusif); satu label ground truth integer (0–8) per citra, dengan keluaran model berupa vektor 9 logits riil unnormalized."),
        ("Status Audit Data (Progres 2)", "4.752 citra RGB (resolusi asli 524 × 524 piksel) telah diaudit secara forensik: 0 berkas korup, 0 berkas kosong. Teridentifikasi 213 pasangan citra dengan dHash ≤ 6 (216 citra unik) berlatar belakang identik."),
        ("Pemisahan Data Anti-Kebocoran", "Diimplementasikan modul src/data_split.py berbasis Stratified Group-Aware Partitioning: Data Latih 70,01% (3.327 citra), Validasi 15,00% (713 citra), dan Uji 14,98% (712 citra). Terverifikasi 100% bebas kebocoran antarsubset (0% data leakage)."),
        ("Format Input dan Caching", "Tensor input Float32 berdimensi [B, 3, 128, 128]. Direncanakan pembuatan cache disk NumPy uint8 terkompresi (~233,5 MB) untuk mengeliminasi bottleneck I/O disk dan menghemat penggunaan RAM pada laptop 8 GB."),
        ("Lingkungan dan Framework", f"Target eksekusi lokal: CPU Laptop Asus (RAM 8 GB). Framework: PyTorch {verification_data['versions']['torch']} CPU build, torchvision {verification_data['versions']['torchvision']}. Beban eksekusi dirancang mandiri tanpa dependensi GPU."),
    ], [1.7, 4.5], "Tabel 3.1.", "Ringkasan Kesiapan Data dan Spesifikasi Teknis Sistem")

    p("Untuk menjamin validitas ilmiah model, 213 pasangan citra berjarak dHash ≤ 6 dikelompokkan ke dalam 50 klaster "
      "independen menggunakan algoritma Connected Components. Seluruh citra dalam satu klaster diisolasi secara ketat ke dalam "
      "satu subset tertentu, sehingga tidak ada pasangan citra berlatar belakang identik yang terpisah antara data latih dan validasi/uji. "
      "Distribusi terperinci per kelas disajikan pada Tabel 3.2.")

    table(["Kelas Material", "Total Sampel", "Train Set (70%)", "Validation Set (15%)", "Test Set (15%)"], [
        ("Cardboard", "461", "323 (70,07%)", "69 (14,97%)", "69 (14,97%)"),
        ("Food Organics", "411", "287 (69,83%)", "62 (15,09%)", "62 (15,09%)"),
        ("Glass", "420", "294 (70,00%)", "63 (15,00%)", "63 (15,00%)"),
        ("Metal", "790", "553 (70,00%)", "119 (15,06%)", "118 (14,94%)"),
        ("Miscellaneous Trash", "495", "347 (70,10%)", "74 (14,95%)", "74 (14,95%)"),
        ("Paper", "500", "350 (70,00%)", "75 (15,00%)", "75 (15,00%)"),
        ("Plastic", "921", "645 (70,03%)", "138 (14,98%)", "138 (14,98%)"),
        ("Textile Trash", "318", "222 (69,81%)", "48 (15,09%)", "48 (15,09%)"),
        ("Vegetation", "436", "306 (70,18%)", "65 (14,91%)", "65 (14,91%)"),
        ("TOTAL", "4.752", "3.327 (70,01%)", "713 (15,00%)", "712 (14,98%)"),
    ], [1.5, 0.9, 1.25, 1.3, 1.25], "Tabel 3.2.", "Distribusi Partisi Data Bebas Kebocoran (Group-Aware Stratified Split)")

    figure("data_split_distribution.png", "Gambar 3.1.", "Distribusi partisi data terstratifikasi bebas kebocoran per kelas material (Train: 3.327, Val: 713, Test: 712).")

    # Subbab 3.2
    heading("3.2 Pemilihan Arsitektur dan Alasan")
    p("Penelitian ini membandingkan dua pendekatan arsitektur jaringan saraf tiruan konvolusional untuk menjawab rumusan masalah "
      "secara komparatif dan obyektif:")
    p("1. MobileNetV3-Small (Transfer Learning Utama): Dipilih secara rasional karena efisiensi komputasi tingginya yang dirancang khusus "
      "untuk perangkat edge dan pemrosesan CPU bersumber daya terbatas [10]. Arsitektur ini mengombinasikan depthwise separable convolution, "
      "inverted residual blocks, serta modul Squeeze-and-Excitation (SE) yang hemat parameter. Seluruh lapisan feature extractor "
      "(927.008 parameter) dibekukan (frozen) dan parameter statistika lapisan BatchNorm dikunci pada mode evaluasi untuk memanfaatkan "
      "representasi fitur visual umum dari dataset berskala besar ImageNet-1K. Hanya lapisan classification head baru yang disesuaikan "
      "dan dilatih secara aktif (600.073 parameter trainable).")
    p("2. CNN Baseline 3-Blok (Titik Acuan dari Nol): Dibangun dari awal tanpa menggunakan bobot pralatih, terdiri atas tiga blok konvolusi "
      "bertingkat (32, 64, dan 128 filter 3×3) dengan aktivasi ReLU dan Max Pooling 2×2, diakhiri dengan Global Adaptive Average Pooling "
      "serta Multi-Layer Perceptron (lapisan linear 64 neuron, Dropout 0,3, dan proyeksi akhir 9 kelas). Model ini berfungsi sebagai "
      "titik acuan pembanding independen untuk mengukur signifikansi perbaikan dari teknik augmentasi dan transfer learning. "
      "Total parameter arsitektur ini adalah 102.089 parameter yang seluruhnya bersifat trainable.")

    table(["Aspek Evaluasi", "CNN Baseline (Dari Nol)", "MobileNetV3-Small (Transfer Learning)"], [
        ("Peran Metodologis", "Titik acuan dasar (baseline reference) tanpa bobot pralatih", "Model transfer learning utama untuk efisiensi edge CPU"),
        ("Feature Extractor", "3 Blok Konvolusi bertingkat (32 -> 64 -> 128 kanal filter)", "ResNet/MobileNet Backbone resmi torchvision MobileNetV3-Small"),
        ("Jumlah Bobot Parameter", f"{verification_data['models']['cnn_baseline']['total_parameters']:,} parameter total (0,39 MB)", f"{verification_data['models']['mobilenet_v3_small']['total_parameters']:,} parameter total (5,83 MB)"),
        ("Status Pelatihan Lapisan", f"{verification_data['models']['cnn_baseline']['trainable_parameters']:,} (100% trainable dari inisialisasi acak)", f"{verification_data['models']['mobilenet_v3_small']['trainable_parameters']:,} trainable (Backbone 927.008 parameter dibekukan)"),
        ("Beban Operasi Aritmetika", f"{profiling_data['models']['cnn_baseline']['macs_millions']} Juta MACs (330,32 Juta FLOPs)", f"{profiling_data['models']['mobilenet_v3_small']['macs_millions']} Juta MACs (37,80 Juta FLOPs; 8,7× lebih efisien)"),
        ("Kelayakan CPU & RAM 8 GB", "Beban memori parameter sangat kecil, aman dieksekusi", "Sangat cepat pada CPU berkat Depthwise Conv; overhead memori minimal"),
        ("Tantangan & Antisipasi", "Rentan terhadap overfitting pada data terbatas; representasi dangkal", "Perbedaan domain visual (ImageNet vs limbah) diatasi adaptasi head baru"),
    ], [1.5, 2.35, 2.35], "Tabel 3.3.", "Perbandingan Karakteristik Arsitektur CNN Baseline dan MobileNetV3-Small")

    # Subbab 3.3
    heading("3.3 Rancangan Input, Output, dan Struktur Model")
    p("Kedua model dirancang untuk memproses tensor citra dengan dimensi spasial yang seragam [B, 3, 128, 128], "
      "namun menerapkan pipeline normalisasi nilai piksel yang berbeda sesuai spesifikasi teknis arsitektur masing-masing:")

    table(["Komponen Kontrak", "Spesifikasi Teknis Operasional"], [
        ("Dimensi Citra Asal", "Format JPEG RGB, resolusi spasial 524 × 524 piksel dengan rentang nilai [0, 255]."),
        ("Dimensi Tensor Masukan", "Tensor Float32 PyTorch berformat [B, 3, 128, 128] (urutan dimensi Batch, Channel, Height, Width)."),
        ("Normalisasi CNN Baseline", "Interpolasi bilinear ke 128 × 128; penskalaan linear piksel [0, 255] menjadi rentang kontinu [0, 1]."),
        ("Normalisasi MobileNetV3", "Bilinear resize 128 × 128; skala [0, 1], standardisasi z-score dengan mean [0.485, 0.456, 0.406] dan deviasi standar [0.229, 0.224, 0.225] standar ImageNet-1K."),
        ("Dimensi Keluaran Model", "Tensor Float32 logits mentah tak ternormalisasi berdimensi [B, 9] tanpa fungsi Softmax di dalam forward pass."),
        ("Target Ground Truth & Loss", "Tensor integer LongTensor berdimensi [B] bernilai 0–8. Menggunakan fungsi kerugian nn.CrossEntropyLoss(logits, target)."),
        ("Inferensi Probabilitas", "Fungsi Softmax(dim=1) hanya diaplikasikan pada saat pengujian/evaluasi; prediksi kelas ditentukan dari argmax(logits)."),
    ], [1.6, 4.6], "Tabel 3.4.", "Rancangan Kontrak Input, Output, dan Fungsi Objektif Pembelajaran")

    figure("cnn_baseline_architecture.png", "Gambar 3.2.", "Diagram arsitektur CNN baseline 3-blok dari tensor input [B, 3, 128, 128] hingga keluaran 9 logits.", width_inch=5.7)
    figure("mobilenetv3_small_architecture.png", "Gambar 3.3.", "Diagram arsitektur MobileNetV3-Small transfer learning dengan backbone dibekukan dan classification head baru.", width_inch=5.7)

    p("Penjelasan Diagram: Pada Gambar 3.2, citra masukan 128×128 direduksi secara bertahap melalui tiga blok konvolusi dan max pooling "
      "menjadi representasi peta fitur 128 kanal berukuran 16×16 piksel. Lapisan Adaptive Average Pooling mereduksi dimensi spasial menjadi "
      "vektor 128 dimensi, dilanjutkan lapisan linear 64 neuron dengan Dropout 0,3 untuk regularisasi, dan lapisan proyeksi akhir menghasilkan "
      "9 logits. Pada Gambar 3.3, feature extractor MobileNetV3-Small mentransformasikan citra masukan menjadi peta fitur 576×4×4. "
      "Head klasifikasi baru memproyeksikannya melalui lapisan linear 1024 neuron dengan aktivasi non-linear Hardswish dan Dropout 0,2 "
      "sebelum menghasilkan 9 nilai logits.")

    figure("data_augmentation_preview.png", "Gambar 3.4.", "Visualisasi pipeline augmentasi citra RealWaste (Original, Random Rotation ±15°, Horizontal Flip, dan Random Zoom 0.9–1.1).", width_inch=5.7)

    # Subbab 3.4
    heading("3.4 Konfigurasi Implementasi Awal dan Profiling Hardware")
    p("Tabel 3.5 merangkum usulan hiperparameter awal yang dikendalikan secara terpusat melalui berkas konfigurasi configs/model_initial.yaml. "
      "Konfigurasi ini dirancang secara realistis untuk menjamin konvergensi yang stabil pada pelatihan CPU laptop.")

    table(["Hiperparameter", "CNN Baseline", "MobileNetV3-Small", "Rasionalisasi Teknis"], [
        ("Loss Function", "CrossEntropyLoss", "CrossEntropyLoss", "Standar objektif untuk klasifikasi multikelas mutual eksklusif."),
        ("Optimizer", "Adam", "AdamW", "Adam cocok untuk konvergensikan bobot acak; AdamW dengan weight decay menjaga kestabilan transfer learning."),
        ("Learning Rate (lr)", "0.001 (1e-3)", "0.0001 (1e-4)", "Learning rate 10× lebih kecil pada MobileNet untuk mencegah destabilisasi fitur pralatih."),
        ("Weight Decay", "0.0", "0.01", "Regularisasi L2 pada AdamW untuk menekan bobot ekstrem pada classification head baru."),
        ("Batch Size Rencana", "32", "16", "Batch size 16 pada MobileNet membatasi konsumsi memori aktivasi tensor pada RAM 8 GB."),
        ("Batas Maksimum Epoch", "30 epoch", "30 epoch", "Batas pelatihan terencana disertai mekanisme Early Stopping (patience=7) berbasis Macro F1 validasi."),
        ("Inisialisasi Bobot", "Inisialisasi Kaiming acak", "IMAGENET1K_V1 (Frozen)", "Memanfaatkan transfer representasi visual umum dari 1,2 juta citra ImageNet."),
    ], [1.3, 1.2, 1.3, 2.4], "Tabel 3.5.", "Rencana Konfigurasi Hiperparameter Awal Model")

    p("Analisis Kelayakan Beban Komputasi pada Perangkat Keras CPU Laptop Asus (RAM 8 GB): "
      "Tabel 3.6 memaparkan hasil profiling teoritis dan verifikasi kelayakan memori. Meskipun MobileNetV3-Small memiliki "
      "parameter 15 kali lebih banyak dibanding CNN baseline, kebutuhan komputasi aritmetikanya (18,90 Juta MACs) justru 8,7 kali lebih "
      "ringan berkat efisiensi depthwise separable convolution. Total konsumsi memori saat pelatihan diproyeksikan di bawah 6% dari "
      "kapasitas RAM 8 GB, memastikan sistem operasi Windows 11 tetap stabil tanpa risiko Out-of-Memory (OOM).")

    table(["Indikator Profiling Hardware", "CNN Baseline", "MobileNetV3-Small", "Keterangan / Implikasi Teknis"], [
        ("Jumlah Parameter Model", "102.089 parameter", "1.527.081 parameter", "CNN Baseline sangat ringkas; MobileNet membawa backbone standar industri."),
        ("Kebutuhan Memori Bobot (MB)", "0,389 MB", "5,825 MB", "Sangat hemat; seluruh bobot model dapat dimuat secara instan ke dalam cache L3 CPU."),
        ("Komputasi MACs per Citra", "165,16 Juta MACs", "18,90 Juta MACs", "MobileNet unggul signifikan (8,7× lebih efisien) berkat arsitektur depthwise separable."),
        ("Komputasi FLOPs per Citra", "330,32 Juta FLOPs", "37,80 Juta FLOPs", "Satu operasi Multiply-Accumulate (MAC) setara dengan 2 operasi FLOPs (Multiply + Add)."),
        ("Estimasi Alokasi Memori Puncak", "361,00 MB", "483,56 MB", "Termasuk dataset cache uint8 (~233,5 MB), memori model, optimizer, dan aktivasi batch."),
        ("Beban terhadap Total RAM 8 GB", "4,41 %", "5,90 %", "Sangat aman; laptop Asus memiliki sisa memori > 94% untuk proses sistem operasi."),
    ], [1.6, 1.2, 1.2, 2.2], "Tabel 3.6.", "Profiling Kompleksitas Komputasi dan Kelayakan Hardware (RAM 8 GB CPU)")

    p("Formulasi Pembobotan Kelas Penyeimbang Ketimpangan (Anti-Leakage Formulation): "
      "Dataset RealWaste memiliki rasio ketimpangan sampel sebesar 2,90 : 1 antara kelas mayoritas (Plastic, 645 sampel latih) "
      "dan kelas minoritas (Textile Trash, 222 sampel latih). Untuk mencegah model bias terhadap kelas mayoritas, "
      "dihitung bobot frekuensi invers yang diturunkan STRICTLY dari subset data latih 3.327 citra (Tabel 3.7).")

    table(["Kelas Material", "Sampel Latih (Train)", "Bobot Invers Mentah", "Bobot Ternormalisasi (Mean = 1.0)"], [
        ("Cardboard", "323", "1,1445", "1,0377"),
        ("Food Organics", "287", "1,2880", "1,1678"),
        ("Glass", "294", "1,2574", "1,1400"),
        ("Metal", "553", "0,6685", "0,6061"),
        ("Miscellaneous Trash", "347", "1,0653", "0,9659"),
        ("Paper", "350", "1,0562", "0,9576"),
        ("Plastic", "645", "0,5731", "0,5196"),
        ("Textile Trash", "222", "1,6652", "1,5098"),
        ("Vegetation", "306", "1,2081", "1,0953"),
    ], [1.8, 1.3, 1.5, 1.6], "Tabel 3.7.", "Formulasi Pembobotan Penyeimbang Kelas (Data Latih)")

    # Subbab 3.5
    heading("3.5 Sampel Baris Program Implementasi Model")
    p("Berikut adalah cuplikan baris program inti dari modul-modul Python yang telah dibangun secara modular dan tervalidasi pada repositori proyek:")

    p("1. Definisi Arsitektur CNN Baseline (src/models/cnn_baseline.py):", bold=True)
    code_block(
        "class CNNBaseline(nn.Module):\n"
        "    def __init__(self, num_classes: int = 9, dropout: float = 0.3):\n"
        "        super().__init__()\n"
        "        self.features = nn.Sequential(\n"
        "            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n"
        "            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n"
        "            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n"
        "        )\n"
        "        self.classifier = nn.Sequential(\n"
        "            nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n"
        "            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),\n"
        "            nn.Linear(64, num_classes),\n"
        "        )\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        return self.classifier(self.features(x))"
    )

    p("2. Definisi MobileNetV3-Small Transfer Learning (src/models/mobilenet_v3.py):", bold=True)
    code_block(
        "class MobileNetV3Small(nn.Module):\n"
        "    def __init__(self, num_classes: int = 9, pretrained: bool = False, freeze_backbone: bool = True):\n"
        "        super().__init__()\n"
        "        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None\n"
        "        self.network = mobilenet_v3_small(weights=weights)\n"
        "        in_features = self.network.classifier[-1].in_features\n"
        "        self.network.classifier[-1] = nn.Linear(in_features, num_classes)\n"
        "        if freeze_backbone:\n"
        "            for p in self.network.features.parameters():\n"
        "                p.requires_grad = False\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        return self.network(x)"
    )

    p("3. Algoritma Pemisahan Data Terstratifikasi Sadar-Kelompok (src/data_split.py):", bold=True)
    code_block(
        "# 1. Bentuk komponen terhubung (connected components) dari 213 pasangan dHash <= 6\n"
        "groups, file_to_group = build_connected_components(inventory_df, near_duplicates_df)\n"
        "# 2. Partisi data latih (70%), validasi (15%), dan uji (15%) berbasis klaster grup\n"
        "splits = partition_stratified_group(inventory_df, groups, target_ratios=(0.70, 0.15, 0.15), seed=42)\n"
        "# 3. Verifikasi integritas: 0% data leakage dan distribusi kelas seimbang\n"
        "verify_split_integrity(splits, file_to_group, total_expected=4752)\n"
        "# Hasil: Train (3.327 sampel), Val (713 sampel), Test (712 sampel)"
    )

    p("Tautan Repositori Proyek dan Bukti Verifikasi Kode:", bold=True)
    p("Repositori GitHub Resmi : https://github.com/tryaannn/klasifikasi-jenis-sampah\n"
      "Notebook Perancangan    : notebooks/02_model_design.ipynb\n"
      "Modul Eksekusi Lokal    : python -m src.verify_models dan python -m unittest tests/test_data_pipeline.py\n"
      "Status Pengujian Unit   : 8 pengujian integritas arsitektur PyTorch + 4 pengujian integritas data pipeline telah LULUS (passed 100%). "
      "Selaras dengan etika dan rubrik akademik, seluruh angka yang disajikan dalam laporan ini merupakan hasil verifikasi komputasi nyata; "
      "tidak ada metrik hasil pelatihan yang dikarang sebelum eksperimen Progres 4 dilaksanakan.")

    p("Referensi Tambahan Progres 3:\n"
      "[10] A. Howard et al., “Searching for MobileNetV3,” in Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1314–1324.\n"
      "[11] PyTorch Team, “CrossEntropyLoss — PyTorch Documentation,” https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html, 2024.\n"
      "[12] Torchvision Maintainers, “MobileNetV3-Small Architecture Specifications,” https://pytorch.org/vision/stable/models/mobilenetv3.html, 2024.\n"
      "[13] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-Balanced Loss Based on Effective Number of Samples,” in Proc. IEEE/CVF CVPR, 2019, pp. 9268–9277.")

    # Checklist Pengumpulan Sesuai Template Dosen
    heading("CHECKLIST EVALUASI SEBELUM PENGUMPULAN (PROGRES 3)")
    table(["Kriteria Rubrik Evaluasi Dosen", "Status Verifikasi Mahasiswa", "Keterangan Bukti Nyata Terlampir"], [
        ("Naskah BAB I dan BAB II dipertahankan utuh", "[ V ] TERPENUHI", "10 halaman Bab I dan II dari Progres 2 dipertahankan tanpa perubahan teks sembarangan."),
        ("Spesifikasi tugas, input, dan target didefinisikan jelas", "[ V ] TERPENUHI", "Klasifikasi 9 kelas material, input [B, 3, 128, 128], target integer [B], logits [B, 9]."),
        ("Rasionalisasi pemilihan arsitektur model diuraikan", "[ V ] TERPENUHI", "Perbandingan komparatif MobileNetV3-Small transfer learning vs CNN Baseline 3-blok dari awal."),
        ("Diagram alur arsitektur dan tabel parameter disertakan", "[ V ] TERPENUHI", "Gambar 3.2, Gambar 3.3, dan Tabel 3.5 menyajikan arsitektur dan parameter terpusat."),
        ("Profiling beban komputasi hardware disajikan", "[ V ] TERPENUHI", "Tabel 3.6 memaparkan analisis FLOPs (18,9M vs 165,2M MACs) dan beban RAM < 6% dari 8 GB."),
        ("Partisi dataset terstratifikasi bebas kebocoran terbukti", "[ V ] TERPENUHI", "Tabel 3.2 & Gambar 3.1: 50 klaster dHash terisolasi, jaminan zero data leakage antarsubset."),
        ("Cuplikan kode implementasi dan tautan repositori aktif", "[ V ] TERPENUHI", "Cuplikan modular cnn_baseline, mobilenet_v3, data_split, dan tautan repositori GitHub."),
        ("Integritas data terjaga tanpa fabrikasi hasil pelatihan", "[ V ] TERPENUHI", "Disiplin ilmiah dipatuhi; metrik akurasi/loss training tidak dikarang sebelum Progres 4."),
    ], [2.2, 1.3, 2.7], "Tabel 3.8.", "Matriks Pemenuhan Kriteria Evaluasi Penugasan (Progres 3)")

    # Save to scratch
    scratch_docx = SCRATCH / "DL_A_KelompokART_Progres3_v2.docx"
    doc.save(str(scratch_docx))
    print(f"Scratch unified report saved successfully: {scratch_docx}")

    # Build standalone BAB_III_Progres3_v2.docx
    doc_standalone = Document()
    for p_elem in doc.paragraphs[72:]:
        p_new = doc_standalone.add_paragraph()
        p_new.text = p_elem.text
        p_new.style = p_elem.style
    scratch_standalone = SCRATCH / "BAB_III_Progres3_v2.docx"
    doc_standalone.save(str(scratch_standalone))
    print(f"Standalone BAB III saved in scratch: {scratch_standalone}")


if __name__ == "__main__":
    build_v2_reports()
