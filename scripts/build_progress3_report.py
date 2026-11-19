"""Build editable BAB III using measured evidence and verbatim source excerpts.

Export with scripts/export_progress3.py, then run scripts/validate_progress3.py.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.config import load_config, CLASS_NAMES


def build() -> None:
    config = load_config()
    evidence = json.loads((ROOT / "results/model_design/model_verification.json").read_text())
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.2677), Inches(11.6929)
    section.top_margin, section.bottom_margin = Inches(.85), Inches(.85)
    section.left_margin, section.right_margin = Inches(1.05), Inches(.95)
    section.footer_distance = Inches(.55)
    for name in ("Normal", "Title", "Heading 1", "Heading 2"):
        style = doc.styles[name]
        # Calibri is installed locally; Aptos from the historical PDF is embedded
        # there but unavailable to the local LibreOffice renderer.
        style.font.name = "Calibri"
        style.font.color.rgb = RGBColor(0, 0, 0)
        fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
        for attribute in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(qn(f"w:{attribute}"), "Calibri")
        for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            fonts.attrib.pop(qn(f"w:{attribute}"), None)
    for border in list(doc.styles.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.1
    normal.paragraph_format.space_after = Pt(6)
    for name in ("Heading 1", "Heading 2"):
        doc.styles[name].font.size = Pt(14 if name == "Heading 1" else 11.5)
        doc.styles[name].paragraph_format.space_before = Pt(10)
        doc.styles[name].paragraph_format.space_after = Pt(8)
    doc.styles["Title"].font.size = Pt(14)
    pgnum = OxmlElement("w:pgNumType")
    pgnum.set(qn("w:start"), "11")
    section._sectPr.append(pgnum)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("Laporan Project Deep Learning | Progres 3 | Halaman ")
    run.font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    def p(text, *, bold=False):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.add_run(text).bold = bold
        return para

    def heading(text):
        doc.add_heading(text, level=2)

    def table(headers, rows, widths=None):
        t = doc.add_table(rows=1, cols=len(headers))
        t.autofit = False
        if widths:
            for column, width in zip(t.columns, widths):
                column.width = Inches(width)
        for i, title in enumerate(headers):
            t.rows[0].cells[i].text = title
        for values in rows:
            cells = t.add_row().cells
            for cell, value in zip(cells, values):
                cell.text = str(value)
        for ri, row in enumerate(t.rows):
            pr = row._tr.get_or_add_trPr()
            no_split = OxmlElement("w:cantSplit")
            pr.append(no_split)
            if ri == 0:
                pr.append(OxmlElement("w:tblHeader"))
            for ci, cell in enumerate(row.cells):
                if widths:
                    cell.width = Inches(widths[ci])
                cell.vertical_alignment = 1
                tcpr = cell._tc.get_or_add_tcPr()
                borders = OxmlElement("w:tcBorders")
                for edge in ("top", "left", "bottom", "right"):
                    e = OxmlElement(f"w:{edge}")
                    for key, val in (("val", "single"), ("sz", "5"), ("color", "D9D9D9")):
                        e.set(qn(f"w:{key}"), val)
                    borders.append(e)
                tcpr.append(borders)
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "E7EEF5" if ri == 0 else "FFFFFF")
                tcpr.append(shading)
                margins = OxmlElement("w:tcMar")
                for edge in ("top", "left", "bottom", "right"):
                    e = OxmlElement(f"w:{edge}")
                    e.set(qn("w:w"), "85")
                    e.set(qn("w:type"), "dxa")
                    margins.append(e)
                tcpr.append(margins)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(2)
                    para.paragraph_format.line_spacing = 1.0
                    for run in para.runs:
                        run.font.size = Pt(9.5)
                        run.bold = ri == 0
        doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return t

    def page():
        doc.add_page_break()

    def figure(filename, caption):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(str(ROOT / "figures" / filename), width=Inches(5.85))
        caption_p = doc.add_paragraph(caption)
        caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in caption_p.runs:
            run.font.size = Pt(9)

    def excerpt(relative, start, end):
        lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
        text = "\n".join(lines[start-1:end])
        para = doc.add_paragraph()
        para.paragraph_format.line_spacing = 1
        para.paragraph_format.space_after = Pt(6)
        para.paragraph_format.keep_together = True
        run = para.add_run(text)
        run.font.name = "Consolas"
        run.font.size = Pt(8)
        excerpts.append({"source": relative, "start_line": start, "end_line": end, "text": text})

    excerpts = []
    doc.add_paragraph("BAB III\nPERANCANGAN DAN IMPLEMENTASI MODEL", style="Title")
    heading("3.1 Ringkasan Studi Kasus dan Kesiapan Data")
    p("Progres 3 melanjutkan studi Kelompok ART pada klasifikasi sembilan material sampah RealWaste. "
      "Tahap ini menghasilkan rancangan dan implementasi awal dua model, verifikasi tensor sintetis, serta "
      "konfigurasi eksperimen. Pelatihan, tuning, dan evaluasi dataset belum dilakukan.")
    table(["Komponen", "Keterangan"], [
        ("Judul project", "Klasifikasi Jenis Sampah Rumah Tangga pada Citra Dunia Nyata Menggunakan CNN: Perbandingan Baseline, Augmentasi, dan Transfer Learning"),
        ("Jenis tugas dan target", "Klasifikasi citra multi-kelas; satu label material per citra, ID 0–8, serta sembilan logits. Probabilitas softmax digunakan saat inferensi."),
        ("Status dataset Progres 2", "4.752 citra RGB, 524 × 524; sembilan kelas. Audit historis: tidak ada citra rusak atau duplikat identik; kandidat near-duplicate telah dicatat, tinjauan manual belum lengkap."),
        ("Kesiapan repositori", "CSV/JSON hasil EDA, empat gambar EDA, notebook 01, dan PDF Progres 2 tersedia. Citra mentah tidak disertakan; tidak diperlukan untuk pengujian arsitektur."),
        ("Input aktual dan rencana", "Aktual pada verifikasi: tensor sintetis [2,3,128,128] pada CPU. Rencana citra: resize bilinear 128 × 128 × 3; cache, normalisasi dan split belum dijalankan."),
        ("Format label", "Integer torch.long berukuran [B], rentang 0–8; pemetaan tetap di src/models/config.py."),
        ("Perangkat dan framework", f"Target proposal tetap CPU dengan RAM 8 GB. Eksekusi ini memakai CPU; Python {evidence['python']}, PyTorch {evidence['versions']['torch']}, torchvision {evidence['versions']['torchvision']}. GPU tidak dipakai; kapasitas training pada RAM 8 GB belum diuji."),
    ], [1.55, 4.7])
    p("Rencana split tetap 70% training, 15% validation, dan 15% testing dengan seed 42, "
      "stratifikasi kelas dan pengelompokan citra mirip. Angka 3.326/713/713 pada BAB II adalah perkiraan; "
      "jumlah aktual harus mengikuti integritas grup. Testing tidak digunakan untuk memilih konfigurasi.")
    p("Penyesuaian jadwal: BAB II masih menempatkan pipeline dan pelatihan pada Progres 3. "
      "Sesuai cakupan tugas terbaru, pekerjaan tersebut dilanjutkan pada Progres 4. BAB I–II "
      "dipertahankan utuh sebagai dokumen historis, tanpa revisi diam-diam.")

    page()
    heading("3.2 Pemilihan Arsitektur dan Alasan")
    p("MobileNetV3-Small dipilih sebagai arsitektur utama transfer learning, sesuai proposal pada BAB I. "
      "Arsitektur ini menggabungkan inverted residual, depthwise convolution, squeeze-and-excitation pada "
      "blok tertentu, dan aktivasi efisien untuk membentuk fitur visual dengan kapasitas terkontrol [10]. "
      "Implementasi memakai pembangun resmi torchvision agar struktur backbone dan jalur pemuatan bobot "
      "dapat dilacak melalui dokumentasi teknis [11].")
    p("Pada rencana pelatihan, fitur awal ImageNet diharapkan membantu adaptasi pada 4.752 citra dengan "
      "tekstur, bentuk, dan pencahayaan yang beragam. Manfaat tersebut masih merupakan hipotesis: "
      "domain ImageNet berbeda dari fasilitas sampah, latar yang serupa dapat menjadi fitur semu, "
      "dan resize 128 × 128 dapat menghilangkan detail material. Belum ada bukti bahwa model ini "
      "mengungguli baseline pada RealWaste.")
    p("CNN baseline tiga blok dibangun dari awal untuk menyediakan pembanding tanpa pengetahuan "
      "pralatih. Kanal 32, 64, dan 128 menambah kapasitas secara bertahap, sedangkan global average "
      "pooling menghindari head yang besar akibat flatten seluruh peta fitur. Dropout 0,3 dipilih "
      "sebagai usulan regularisasi; dampaknya baru dapat ditentukan melalui validasi setelah training.")
    table(["Aspek", "CNN baseline", "MobileNetV3-Small"], [
        ("Peran", "Acuan dari awal", "Transfer learning utama"),
        ("Fitur", "Tiga blok Conv–ReLU–MaxPool", "Backbone resmi torchvision"),
        ("Bobot rencana", "Inisialisasi acak", "IMAGENET1K_V1"),
        ("Verifikasi Progres 3", "Bobot acak; semua parameter trainable", "Bobot acak; features frozen, classifier trainable"),
        ("Keterbatasan", "Belum memiliki fitur pralatih; kapasitas kecil dapat membatasi representasi", "Belum memuat bobot ImageNet; domain dan resolusi input berbeda dari pralatih"),
    ], [1.25, 2.5, 2.5])
    p("Kedua model menjadi dasar perbandingan terkontrol: kontrol tanpa augmentasi/pembobotan, "
      "varian augmentasi, varian pembobotan kelas, dan transfer learning. Seluruh varian harus memakai "
      "split yang sama. Pembobotan kelas tidak diaktifkan secara otomatis, agar pengaruhnya dapat dipisahkan.")
    p("Referensi arsitektur: [10] A. Howard et al., Searching for MobileNetV3, ICCV 2019, "
      "doi:10.1109/ICCV.2019.00140 (referensi historis BAB II). [11] Torchvision, mobilenet_v3_small, "
      "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html "
      "(diakses 18 November 2026).")

    page()
    heading("3.3 Rancangan Input, Output, dan Struktur Model")
    table(["Spesifikasi", "Rancangan"], [
        ("Citra asal", "RGB 524 × 524 × 3; satu label material per citra"),
        ("Input model", "128 × 128 × 3, float32, format PyTorch NCHW [B,3,128,128]"),
        ("Input CNN", "Resize bilinear; penskalaan piksel 0–255 menjadi [0,1]"),
        ("Input MobileNet", "Resize bilinear 128 × 128; skala [0,1], lalu mean [0,485;0,456;0,406] dan std [0,229;0,224;0,225]"),
        ("Output kedua model", "Logits [B,9]; target integer torch.long [B], ID 0–8"),
        ("Aktivasi keluaran", "Linear tanpa softmax di dalam model; softmax(dim=1) di luar model saat inferensi; prediksi kelas = argmax logits"),
    ], [1.6, 4.65])
    table(["ID", "Label kelas asli"], list(enumerate(CLASS_NAMES)), [.65, 5.6])
    p("Citra terlebih dahulu direncanakan di-resize dan dinormalisasi sesuai kebijakan arsitektur sebelum "
      "dibentuk menjadi batch NCHW. CNN mengekstraksi fitur lewat tiga blok konvolusi dan pooling, "
      "sedangkan MobileNet memakai feature extractor resminya. Adaptive average pooling merangkum "
      "peta fitur, kemudian classification head memetakannya menjadi sembilan logits. CrossEntropyLoss "
      "menerima logits dan target integer langsung untuk objektif multikelas. Saat probabilitas diperlukan "
      "untuk inferensi atau penyajian, softmax diterapkan di luar model sehingga total probabilitas tiap "
      "citra bernilai satu.")
    p("Istilah sparse categorical crossentropy pada BAB II dipadankan dengan nn.CrossEntropyLoss "
      "PyTorch untuk label integer [12]. Perubahan istilah ini tidak mengubah tugas klasifikasi. "
      "Transform bawaan bobot ImageNet memakai resize 256 dan crop 224; proyek tetap memakai input "
      "128 sesuai proposal dan mengambil mean–std ImageNet secara eksplisit [11]. Pengaruh adaptasi "
      "resolusi belum dievaluasi.")

    page()
    p("Struktur CNN baseline", bold=True)
    figure("cnn_baseline_architecture.png", "Gambar 3.1 Arsitektur CNN baseline dan konversi probabilitas saat inferensi.")
    table(["Tahap", "Bentuk tensor keluaran"], [
        ("Blok 1: Conv 3→32, ReLU, MaxPool", "[B,32,64,64]"),
        ("Blok 2: Conv 32→64, ReLU, MaxPool", "[B,64,32,32]"),
        ("Blok 3: Conv 64→128, ReLU, MaxPool", "[B,128,16,16]"),
        ("AdaptiveAvgPool, Flatten, Linear 128→64", "[B,128,1,1] → [B,128] → [B,64]"),
        ("ReLU, Dropout 0,3, Linear 64→9", "[B,9] logits"),
    ], [3.35, 2.9])
    p("Seluruh konvolusi menggunakan kernel 3 × 3, stride 1, padding 1 dan bias. MaxPool memakai "
      "kernel/stride 2. Bentuk tensor pada diagram diturunkan dari forward hooks implementasi aktual; "
      "SVG dan generator diagram disertakan untuk reproduksi.")

    page()
    p("Struktur MobileNetV3-Small", bold=True)
    figure("mobilenetv3_small_architecture.png", "Gambar 3.2 MobileNetV3-Small dengan classification head sembilan kelas.")
    p("Pada input 128 × 128, feature extractor menghasilkan [B,576,4,4]. AdaptiveAvgPool2d(1) "
      "dan flatten menghasilkan 576 fitur, dilanjutkan Linear 576→1024, Hardswish, dropout 0,2, "
      "dan Linear 1024→9. Seluruh classifier tetap trainable, bukan hanya linear terakhir.")
    p("Pembekuan meliputi requires_grad=False pada features serta eval mode untuk blok yang "
      "dibekukan agar running mean/variance BatchNorm tidak berubah. Metode set_trainable_feature_blocks "
      "membuka N blok terakhir untuk fine-tuning berikutnya; optimizer harus dibuat ulang setelah "
      "kebijakan ini diubah. Tidak ada fine-tuning yang dijalankan pada Progres 3.")
    p("Status bobot dibedakan secara eksplisit: verifikasi memakai weights=None; konfigurasi "
      "transfer learning Progres 4 merencanakan IMAGENET1K_V1; pemuatan bobot ImageNet belum dilakukan. "
      "Backbone acak yang dibekukan hanya digunakan untuk uji struktur dan bukan sebagai eksperimen transfer learning.")

    page()
    heading("3.4 Konfigurasi Implementasi Awal")
    p("Tabel 3.1 disusun dari configs/model_initial.yaml. Nilai berikut adalah usulan awal, belum "
      "dituning dan belum menunjukkan konfigurasi optimal. Seed inisialisasi dan verifikasi adalah 42.")
    cnn, mob = (config["models"][k] for k in ("cnn_baseline", "mobilenet_v3_small"))
    table(["Komponen", "CNN baseline", "MobileNetV3-Small"], [
        ("Loss function", cnn["loss"], mob["loss"]),
        ("Optimizer", cnn["optimizer"], mob["optimizer"]),
        ("Learning rate", cnn["learning_rate"], mob["learning_rate"]),
        ("Batch size rencana", cnn["batch_size"], mob["batch_size"]),
        ("Maksimum epoch rencana", cnn["max_epochs_planned"], mob["max_epochs_planned"]),
        ("Weight decay", cnn["weight_decay"], mob["weight_decay"]),
        ("Monitoring Progres 4", "Loss, akurasi, precision/recall/F1 makro validasi", "Loss, akurasi, precision/recall/F1 makro validasi"),
        ("Bobot / freeze rencana", "Tanpa pretrained; semua parameter trainable", "IMAGENET1K_V1; features frozen; classifier trainable"),
        ("Bobot verifikasi aktual", "Acak", "Acak (tanpa unduhan)"),
    ], [1.6, 2.325, 2.325])
    p("Adam dengan learning rate 0,001 menjadi titik awal pembanding dari awal. AdamW dengan learning "
      "rate 0,0001 dan weight decay 0,01 disiapkan untuk adaptasi transfer learning yang lebih konservatif. "
      "Batch size MobileNet 16 dipilih lebih kecil untuk membatasi memori; CNN memakai 32. Maksimum "
      "30 epoch adalah batas rencana, bukan jumlah epoch yang telah dijalankan. Kelayakan memori dan "
      "waktu pelatihan perlu diverifikasi sebelum eksperimen.")
    p("Augmentasi (flip, rotasi ±15°, zoom 0,9–1,1 dan variasi pencahayaan ringan) serta pembobotan "
      "kelas tetap menjadi eksperimen Progres 4. Keduanya dinonaktifkan pada konfigurasi kontrol. "
      "Statistik bobot kelas dihitung hanya dari training. Testing tetap terpisah sampai konfigurasi akhir.")
    table(["Hasil verifikasi aktual pada CPU", "CNN baseline", "MobileNetV3-Small"], [
        ("Input / output", "[2,3,128,128] / [2,9]", "[2,3,128,128] / [2,9]"),
        ("Parameter total", evidence["models"]["cnn_baseline"]["total_parameters"], evidence["models"]["mobilenet_v3_small"]["total_parameters"]),
        ("Parameter trainable", evidence["models"]["cnn_baseline"]["trainable_parameters"], evidence["models"]["mobilenet_v3_small"]["trainable_parameters"]),
        ("Finite / deterministik eval", "Ya / Ya", "Ya / Ya"),
        ("Backbone features frozen", "Tidak diterapkan", "Ya; classifier tetap trainable"),
    ], [2.4, 1.925, 1.925])
    p("Delapan pengujian otomatis lulus. Notebook 02 dieksekusi menggunakan tensor sintetis. Bukti "
      "tersimpan di results/model_design/ dan notebook; hasil ini memverifikasi integritas struktur, "
      "bukan akurasi atau kemampuan generalisasi. Tidak ada metrik pelatihan atau evaluasi dataset.")

    page()
    heading("3.5 Sampel Baris Program Implementasi Model")
    p("Cuplikan berikut diambil langsung dari berkas sumber saat dokumen dibangun. CNN menggunakan "
      "torch dan nn; blok fitur serta head berada dalam satu kelas yang dapat dipakai ulang.")
    excerpt("src/models/cnn_baseline.py", 4, 7)
    excerpt("src/models/cnn_baseline.py", 13, 23)
    p("MobileNet menggunakan pembangun resmi torchvision dan mengganti linear terakhir menjadi "
      "sembilan keluaran. pretrained=False pada verifikasi memastikan tidak ada unduhan bobot.")
    excerpt("src/models/mobilenet_v3.py", 4, 6)
    excerpt("src/models/mobilenet_v3.py", 15, 25)
    p("Komponen loss dan optimizer dibuat dari konfigurasi pusat; hanya parameter trainable yang "
      "dimasukkan. Pembuatan optimizer tidak menjalankan langkah pembaruan parameter.")
    excerpt("src/verify_models.py", 24, 30)
    p("Bukti kode: https://github.com/tryaannn/klasifikasi-jenis-sampah; penambahan Progres 3 "
      "berada pada working tree lokal dan belum di-commit/push. Notebook: notebooks/02_model_design.ipynb. "
      "Verifikasi: python -m src.verify_models; pengujian: python -m pytest -q. Pelatihan penuh, "
      "backward pass, optimizer step, dan evaluasi akhir belum dilakukan.")
    p("[12] PyTorch, CrossEntropyLoss, https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html "
      "(diakses 18 November 2026). Pemetaan objective menggunakan logits dan label integer sesuai "
      "dokumentasi; softmax diterapkan hanya untuk probabilitas inferensi.")
    path = ROOT / "laporan/BAB_III_Progres3.docx"
    doc.save(path)
    (ROOT / "results/model_design/report_code_excerpts.json").write_text(
        json.dumps(excerpts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
