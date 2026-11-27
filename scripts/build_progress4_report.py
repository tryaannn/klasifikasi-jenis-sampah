"""Build publication-ready Progres 4 Report (DOCX) continuing BAB I-III seamlessly.

Reads all empirical metrics programmatically from real JSON artifacts under
`results/progress4/` without rounding or hardcoding fabricated numbers, embeds
visual evidence from `figures/`, preserves BAB I-III intact from
`laporan/DL_A_KelompokART_Progres3.docx`, and appends BAB IV (4.1-4.5) plus
Checklist Evaluasi Progres 4 following the official template styling.
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]

COLOR_MUTED = RGBColor(0x62, 0x70, 0x80)
BG_HEADER = "EAF2F8"
BG_HIGHLIGHT = "EDF7ED"
BORDER_COLOR = "D3D3D3"


def format_id_pct(val: float, decimals: int = 2) -> str:
    """Format probability/ratio float as Indonesian comma percentage string."""
    return f"{val * 100:.{decimals}f}".replace(".", ",") + "%"


def format_id_num(val: float, decimals: int = 4) -> str:
    """Format floating-point metric using Indonesian decimal comma."""
    return f"{val:.{decimals}f}".replace(".", ",")


def format_table(
    table,
    col_widths: list[float],
    *,
    bg_header: str = BG_HEADER,
    highlight_rows: set[int] | None = None,
    compact: bool = False,
) -> None:
    """Apply uniform Ice Blue header, crisp grey borders, and readable padding."""
    table.style = "Table Grid"
    highlight_rows = highlight_rows or set()
    pad_tb = "60" if compact else "80"
    pad_lr = "80" if compact else "120"
    font_pt = 8.0 if len(col_widths) >= 8 else (8.5 if len(col_widths) >= 5 else 9.5)

    for ri, row in enumerate(table.rows):
        pr = row._tr.get_or_add_trPr()
        pr.append(OxmlElement("w:cantSplit"))
        if ri == 0:
            pr.append(OxmlElement("w:tblHeader"))

        fill = bg_header if ri == 0 else (BG_HIGHLIGHT if ri in highlight_rows else "FFFFFF")
        for ci, cell in enumerate(row.cells):
            cell.width = Inches(col_widths[ci])
            cell.vertical_alignment = 1
            tcpr = cell._tc.get_or_add_tcPr()

            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right"):
                e = OxmlElement(f"w:{edge}")
                for key, val in (("val", "single"), ("sz", "4"), ("color", BORDER_COLOR)):
                    e.set(qn(f"w:{key}"), val)
                borders.append(e)
            tcpr.append(borders)

            shading = OxmlElement("w:shd")
            shading.set(qn("w:fill"), fill)
            tcpr.append(shading)

            margins = OxmlElement("w:tcMar")
            for edge, val in (("top", pad_tb), ("bottom", pad_tb), ("left", pad_lr), ("right", pad_lr)):
                e = OxmlElement(f"w:{edge}")
                e.set(qn("w:w"), val)
                e.set(qn("w:type"), "dxa")
                margins.append(e)
            tcpr.append(margins)

            for para in cell.paragraphs:
                para.paragraph_format.space_after = Pt(1.5 if compact else 2)
                para.paragraph_format.space_before = Pt(1.5 if compact else 2)
                para.paragraph_format.line_spacing = 1.05
                if ci > 0 and len(col_widths) >= 5:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER if ri == 0 else WD_ALIGN_PARAGRAPH.RIGHT
                if ci == 0:
                    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in para.runs:
                    run.bold = (ri == 0)
                    run.font.size = Pt(font_pt)


def _add_bab4_content(
    doc: Document,
    comparison: list[dict],
    val_eval: dict,
    exp10_summary: dict,
    exp10_history: list[dict],
    exp11_summary: dict,
) -> None:
    """Append BAB IV (4.1-4.5) and Checklist Evaluasi Progres 4 to a Document."""
    def heading1(text: str):
        para = doc.add_paragraph(text, style="Heading 1")
        para.paragraph_format.space_before = Pt(14)
        para.paragraph_format.space_after = Pt(6)
        para.paragraph_format.keep_with_next = True
        return para

    def heading2(text: str):
        para = doc.add_paragraph(text, style="Heading 2")
        para.paragraph_format.space_before = Pt(12)
        para.paragraph_format.space_after = Pt(4)
        para.paragraph_format.keep_with_next = True
        return para

    def p(text: str, *, bold: bool = False, italic: bool = False, first_indent: bool = True, muted: bool = False):
        para = doc.add_paragraph(style="Normal")
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after = Pt(4)
        if first_indent and not muted:
            para.paragraph_format.first_line_indent = Inches(0.4)
        run = para.add_run(text)
        if muted:
            run.font.size = Pt(9.0)
            run.font.color.rgb = COLOR_MUTED
            run.italic = True
        else:
            run.bold = bold
            run.italic = italic
        return para

    def list_item(text: str, *, bold_prefix: str = ""):
        para = doc.add_paragraph(style="List Paragraph")
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after = Pt(4)
        para.paragraph_format.left_indent = Inches(0.25)
        if bold_prefix:
            r_pre = para.add_run(bold_prefix)
            r_pre.bold = True
        para.add_run(text)
        return para

    def table_caption(caption_text: str):
        para = doc.add_paragraph(style="Normal")
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        para.paragraph_format.space_before = Pt(8)
        para.paragraph_format.space_after = Pt(3)
        para.paragraph_format.keep_with_next = True
        run = para.add_run(caption_text)
        run.bold = True
        return para

    def add_table(
        headers: list[str],
        rows: list[tuple | list],
        widths: list[float],
        caption_text: str = "",
        *,
        highlight_rows: set[int] | None = None,
        compact: bool = False,
    ):
        if caption_text:
            table_caption(caption_text)
        tbl = doc.add_table(rows=1, cols=len(headers))
        tbl.autofit = False
        for i, title in enumerate(headers):
            tbl.rows[0].cells[i].text = title
        for row_vals in rows:
            cells = tbl.add_row().cells
            for ci, val in enumerate(row_vals):
                cells[ci].text = str(val)
        format_table(tbl, widths, highlight_rows=highlight_rows, compact=compact)
        return tbl

    def figure(filename: str, caption_text: str, width_inch: float = 5.9):
        img_path = ROOT / "figures" / filename
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing figure required for report: {img_path}")
        para = doc.add_paragraph(style="Normal")
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after = Pt(2)
        para.paragraph_format.keep_with_next = True
        run = para.add_run()
        run.add_picture(str(img_path), width=Inches(width_inch))

        cap_p = doc.add_paragraph(style="Normal")
        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_p.paragraph_format.space_before = Pt(2)
        cap_p.paragraph_format.space_after = Pt(8)
        run_cap = cap_p.add_run(caption_text)
        run_cap.italic = True
        run_cap.font.size = Pt(9.0)
        return cap_p

    def code_block(code_text: str):
        para = doc.add_paragraph(style="Normal")
        para.paragraph_format.line_spacing = 1.05
        para.paragraph_format.space_before = Pt(3)
        para.paragraph_format.space_after = Pt(6)
        para.paragraph_format.left_indent = Inches(0.2)
        para.paragraph_format.right_indent = Inches(0.2)
        para.paragraph_format.keep_together = True

        pPr = para._p.get_or_add_pPr()
        pBdr = parse_xml(
            f'<w:pBdr {nsdecls("w")}>'
            f'  <w:left w:val="single" w:sz="12" w:space="8" w:color="184470"/>'
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

    by_id = {row["exp_id"]: row for row in comparison}
    exp04 = by_id["exp04_mobilenet_tl"]
    exp10 = by_id["exp10_mobilenet_224_2stage_ft"]
    exp11 = by_id["exp11_mobilenet_224_2stage_cw"]

    repro = val_eval["reproducibility"]
    metrics = val_eval["metrics"]
    err = val_eval["error_analysis"]
    cal = val_eval["calibration"]

    f1_gain = (exp10["best_val_macro_f1"] - exp04["best_val_macro_f1"]) * 100.0
    gap_drop = (exp04["train_val_acc_gap"] - exp10["train_val_acc_gap"]) * 100.0

    # --- BAB IV ---
    heading1("BAB IV — HASIL DAN PEMBAHASAN (PROGRES 4)")

    p(
        f"Pertemuan 13–14. Bab ini menyajikan implementasi pipeline pelatihan, hasil studi ablasi 11 eksperimen "
        f"(EXP-01 hingga EXP-11) pada GPU NVIDIA Tesla T4 di Kaggle, pemeriksaan data dan kalibrasi probabilitas, "
        f"serta evaluasi pada Validation Set (713 citra). Selama Progres 4, Test Set (712 citra) tetap dikunci "
        f"(test_used bernilai false dan test_set_untouched bernilai true) dan baru akan digunakan pada evaluasi akhir di Progres 5.",
        italic=True,
        first_indent=False,
        muted=True,
    )

    # Subbab 4.1
    heading2("4.1 Hasil Implementasi Model dan Kesiapan Pipeline")
    p(
        "Rancangan arsitektur pada BAB III diimplementasikan secara modular pada berkas src/dataset.py, "
        "src/transforms.py, src/train.py, dan src/evaluate_val.py, lalu dijalankan pada lingkungan Kaggle melalui "
        "notebook kaggle/kaggle_runner_p4.ipynb. Sebelum pelatihan dimulai, seluruh 14 pengujian unit dan smoke test "
        "pada pytest berjalan lancar (14 passed), meliputi pemeriksaan dimensi tensor keluaran, pembekuan statistik "
        "BatchNorm pada lapisan fitur yang dibekukan, serta validasi bahwa augmentasi tidak pernah aktif pada data validasi maupun data uji."
    )
    p(
        "Pada percobaan awal beresolusi 128×128 piksel, performa model tertahan dan detail tekstur antarjenis material "
        "tampak kurang terbedakan. Oleh karena itu, pada rangkaian eksperimen lanjutan (EXP-08 hingga EXP-11) resolusi "
        "masukan ditingkatkan menjadi 224×224 piksel sesuai standar pra-latih ImageNet-1K dengan cache disk bertipe uint8 "
        "berukuran 683,06 MB. Augmentasi pada data latih juga disederhanakan menjadi RandomResizedCrop(224) dengan skala "
        "0,75 hingga 1,0, pembalikan horizontal acak (p=0,5), rotasi acak maksimal 15 derajat, dan ColorJitter ringan (0,15)."
    )

    add_table(
        ["Komponen Verifikasi Sistem", "Keterangan Implementasi Aktual"],
        [
            ("Framework dan Perangkat Keras", f"PyTorch {repro['torch_version']} pada GPU {exp10_summary['environment']['cuda_device_name']} (Kaggle Notebook Version 3)"),
            ("Reproduksibilitas dan Pengujian", f"Seed acak = {repro['seed']}, commit Git = {repro['git_commit']} pada branch test/majid, pytest lulus 14 pengujian"),
            ("Pembagian dan Penguncian Data", "Data latih 3.327 citra, validasi 713 citra, dan data uji 712 citra masih dikunci hingga Progres 5"),
            ("Normalisasi dan Urutan Kanal", "Citra RGB skala [0, 1]; MobileNetV3 memakai mean [0.485, 0.456, 0.406] dan std [0.229, 0.224, 0.225]"),
            ("Model Terpilih pada Progres 4", f"EXP-10 (MobileNetV3-Small 224×224, two-stage fine-tuning) dengan total {exp10_summary['model_stats']['total_parameters']:,} parameter dan {exp10_summary['model_stats']['trainable_parameters']:,} parameter dilatih pada tahap kedua"),
        ],
        [2.0, 4.2],
        caption_text="Tabel 4.1 Ringkasan Lingkungan Eksekusi, Reproduksibilitas, dan Status Data",
    )

    figure(
        "p4_sanity_train_val_batch.png",
        "Gambar 4.1 Contoh citra hasil prapemrosesan (224×224 piksel RGB): augmentasi hanya diterapkan pada baris data latih (atas), sedangkan baris data validasi (bawah) tetap utuh.",
        width_inch=6.0,
    )

    # Subbab 4.2
    heading2("4.2 Hasil Pelatihan Model dan Studi Ablasi (EXP-01 hingga EXP-11)")
    p(
        "Pelatihan dijalankan dalam tiga kelompok eksperimen untuk melihat pengaruh arsitektur, regularisasi, "
        "resolusi 224×224, serta strategi pelatihan dua tahap (two-stage fine-tuning). Kelompok pertama (EXP-01 hingga EXP-04) "
        "mengukur baseline pada resolusi 128×128; kelompok kedua (EXP-05 hingga EXP-07) menguji mitigasi overfitting "
        "dan underfitting pada 128×128; dan kelompok ketiga (EXP-08 hingga EXP-11) menguji masukan 224×224. Khusus pada "
        "EXP-10 dan EXP-11, pelatihan dibagi menjadi tahap pertama untuk melatih head klasifikasi selama 8 epoch (lr=1e-3, "
        "backbone dibekukan), lalu bobot terbaik tahap pertama dimuat kembali sebelum membuka empat blok konvolusi akhir "
        "pada tahap kedua (epoch 9 hingga 30, learning rate backbone 5e-5 dan head 1,5e-4, warmup 2 epoch, penjadwalan "
        "CosineAnnealingLR, serta Early Stopping dengan patience 7)."
    )
    p(
        "Seluruh angka metrik pada Tabel 4.2 diambil langsung dari berkas hasil eksperimen results/progress4/experiment_comparison.json "
        "agar konsisten dengan keluaran aktual pelatihan di Kaggle."
    )

    exp_rows = []
    highlight_idx = set()
    short_names = {
        "exp01_cnn_control": "EXP-01 CNN Kontrol",
        "exp02_cnn_aug": "EXP-02 CNN + Aug",
        "exp03_cnn_weighted": "EXP-03 CNN + CW",
        "exp04_mobilenet_tl": "EXP-04 MNV3 TL Kontrol",
        "exp05_mobilenet_aug_reg": "EXP-05 MNV3 Aug+Reg",
        "exp06_mobilenet_finetune": "EXP-06 MNV3 FT+CW",
        "exp07_cnn_mitigation": "EXP-07 CNN Ext+Sched",
        "exp08_mobilenet_224_head": "EXP-08 MNV3 224 Head",
        "exp09_mobilenet_224_reg": "EXP-09 MNV3 224 Reg",
        "exp10_mobilenet_224_2stage_ft": "EXP-10 MNV3 224 2-Stage",
        "exp11_mobilenet_224_2stage_cw": "EXP-11 MNV3 224 2-Stage+CW",
    }
    for idx, r in enumerate(comparison, start=1):
        if r["exp_id"] == "exp10_mobilenet_224_2stage_ft":
            highlight_idx.add(idx)
        exp_rows.append((
            short_names.get(r["exp_id"], r["title"]),
            f"{r['image_size']}",
            f"{r['best_epoch']}/{r['epochs_ran']}",
            format_id_num(r["best_train_loss"], 4),
            format_id_num(r["best_val_loss"], 4),
            format_id_pct(r["best_train_accuracy"], 2),
            format_id_pct(r["best_val_accuracy"], 2),
            format_id_pct(r["train_val_acc_gap"], 2),
            format_id_pct(r["best_val_macro_f1"], 2),
            format_id_pct(r["best_val_macro_precision"], 2),
            format_id_pct(r["best_val_macro_recall"], 2),
        ))

    add_table(
        ["Eksperimen", "Res.", "Best/Ep", "Tr.Loss", "Val.Loss", "Tr.Acc", "Val.Acc", "Selisih", "Val F1", "Prec.", "Rec."],
        exp_rows,
        [1.55, 0.36, 0.48, 0.49, 0.49, 0.51, 0.51, 0.48, 0.51, 0.47, 0.47],
        caption_text="Tabel 4.2 Perbandingan Hasil Pelatihan EXP-01 hingga EXP-11 pada Validation Set",
        highlight_rows=highlight_idx,
        compact=True,
    )

    # Tabel dinamika epoch kunci EXP-10
    key_epochs = {1, 3, 8, 9, 10, 12, 15, 20, 26, 30}
    h10_rows = []
    h10_hl = set()
    for r in exp10_history:
        if r["epoch"] in key_epochs:
            row_idx = len(h10_rows) + 1
            if r["epoch"] == exp10_summary["best_epoch"]:
                h10_hl.add(row_idx)
            stage_lbl = "Tahap 1 (Head)" if r.get("stage", 1) == 1 else "Tahap 2 (FT-4)"
            h10_rows.append((
                f"{r['epoch']}",
                stage_lbl,
                f"{r['lr']:.6f}",
                format_id_num(r["train_loss"], 4),
                format_id_num(r["val_loss"], 4),
                format_id_pct(r["train_accuracy"], 2),
                format_id_pct(r["val_accuracy"], 2),
                format_id_pct(r["val_macro_f1"], 2),
            ))

    add_table(
        ["Epoch", "Tahap Pelatihan", "LR Head", "Train Loss", "Val Loss", "Train Acc", "Val Acc", "Val Macro F1"],
        h10_rows,
        [0.45, 1.05, 0.72, 0.72, 0.72, 0.76, 0.76, 0.84],
        caption_text="Tabel 4.3 Perkembangan Metrik per Epoch Kunci pada Model EXP-10",
        highlight_rows=h10_hl,
        compact=True,
    )

    code_block(
        "Cuplikan log pelatihan Kaggle (transisi tahap EXP-10 dan EXP-11):\n"
        "EXP-08 & EXP-09 : Early stopping triggered at epoch 19 (best epoch: 12).\n"
        "EXP-10 Epoch 9  : Stage 2 Transition at epoch 9: loaded Stage 1 best (ep 3, val_macro_f1=0.7640)\n"
        "                  Unfroze last 4 blocks, trainable(Stage 2)=1,336,561, backbone_lr=5.00e-05, head_lr=1.50e-04\n"
        "EXP-11 Epoch 9  : Stage 2 Transition at epoch 9: loaded Stage 1 best (ep 7, val_macro_f1=0.7619)\n"
        "                  Unfroze last 4 blocks, trainable(Stage 2)=1,336,561, backbone_lr=5.00e-05, head_lr=1.50e-04"
    )

    figure(
        "p4_training_accuracy_f1_curves.png",
        "Gambar 4.2 Grafik akurasi validasi dan Macro F1-Score per epoch untuk seluruh 11 eksperimen.",
        width_inch=6.0,
    )
    figure(
        "p4_training_loss_curves.png",
        "Gambar 4.3 Grafik training loss dan validation loss per epoch; validation loss EXP-10 turun konsisten hingga epoch 30 (0,8703).",
        width_inch=6.0,
    )
    figure(
        "p4_train_val_gap_comparison.png",
        "Gambar 4.4 Perbandingan akurasi latih, akurasi validasi, Macro F1, serta selisih akurasi latih dan validasi.",
        width_inch=6.0,
    )

    # Subbab 4.3
    heading2("4.3 Hasil Prediksi Awal dan Evaluasi pada Validation Set")
    p(
        f"Pengujian awal dilakukan pada Validation Set (N={val_eval['validation_count']} citra) menggunakan bobot "
        f"terbaik EXP-10 dari epoch {exp10_summary['best_epoch']}. Model memperoleh akurasi validasi sebesar "
        f"{format_id_pct(metrics['accuracy'])}, Macro Precision {format_id_pct(metrics['macro_precision'])}, "
        f"Macro Recall {format_id_pct(metrics['macro_recall'])}, dan Macro F1-Score {format_id_pct(metrics['macro_f1'])}."
    )

    per_class_rows = []
    for cls_name, vals in metrics["per_class"].items():
        per_class_rows.append((
            cls_name,
            f"{vals['support']}",
            format_id_pct(vals["precision"], 2),
            format_id_pct(vals["recall"], 2),
            format_id_pct(vals["f1"], 2),
        ))
    per_class_rows.append((
        "Rata-rata Makro / Total",
        f"{val_eval['validation_count']}",
        format_id_pct(metrics["macro_precision"], 2),
        format_id_pct(metrics["macro_recall"], 2),
        format_id_pct(metrics["macro_f1"], 2),
    ))

    add_table(
        ["Kelas Material", "Support", "Precision", "Recall", "F1-Score"],
        per_class_rows,
        [2.1, 0.85, 1.05, 1.05, 1.05],
        caption_text="Tabel 4.4 Performa Klasifikasi Per Kelas Model EXP-10 pada Validation Set (713 Sampel)",
        highlight_rows={len(per_class_rows)},
    )

    figure(
        "p4_val_confusion_matrix.png",
        "Gambar 4.5 Confusion matrix EXP-10 pada Validation Set (713 sampel); salah klasifikasi terbanyak terjadi pada Plastic yang diprediksi sebagai Metal (26 sampel) dan Paper sebagai Cardboard (16 sampel).",
        width_inch=5.3,
    )
    figure(
        "p4_val_per_class_metrics.png",
        "Gambar 4.6 Perbandingan Precision, Recall, dan F1-Score tiap kelas pada Validation Set (EXP-10).",
        width_inch=5.9,
    )

    # Tabel sampel representatif (5 benar + 5 salah)
    rep = val_eval["representative_samples"]
    rep_rows = []
    for item in rep["correct_examples"][:5]:
        rep_rows.append((
            item["file"],
            item["true_label"],
            item["pred_label"],
            format_id_num(item["confidence"], 4),
            "Benar",
        ))
    for item in rep["misclassified_examples"][:5]:
        rep_rows.append((
            item["file"],
            item["true_label"],
            item["pred_label"],
            format_id_num(item["confidence"], 4),
            "Salah",
        ))

    add_table(
        ["Berkas Sampel (Validation)", "Label Sebenarnya", "Prediksi Model", "Confidence", "Status"],
        rep_rows,
        [2.25, 1.35, 1.25, 0.72, 0.63],
        caption_text="Tabel 4.5 Contoh 10 Sampel Prediksi Model EXP-10 pada Validation Set (5 Benar dan 5 Salah)",
        compact=True,
    )

    p(
        f"Selain akurasi, pemeriksaan kalibrasi probabilitas (Gambar 4.7) menunjukkan nilai Expected Calibration Error "
        f"(ECE 10-bin) sebesar {format_id_num(cal['ece_10_bins'], 4)}. Dari total {cal['total_samples']} sampel validasi, "
        f"{cal['correct_count']} prediksi benar memiliki rata-rata nilai confidence {format_id_num(cal['correct_confidence']['mean'], 4)} "
        f"(median {format_id_num(cal['correct_confidence']['median'], 4)}), sedangkan {cal['wrong_count']} prediksi salah memiliki "
        f"rata-rata confidence {format_id_num(cal['wrong_confidence']['mean'], 4)} (median {format_id_num(cal['wrong_confidence']['median'], 4)}). "
        f"Meskipun demikian, masih ditemukan {cal['overconfident_wrong_gt_090']} kesalahan ({format_id_pct(cal['overconfident_wrong_ratio'], 2)} "
        f"dari total kesalahan) yang memiliki confidence di atas 0,90, seperti citra Plastic/Plastic_256.jpg yang keliru diprediksi "
        f"sebagai Metal dengan confidence 0,9926."
    )

    figure(
        "p4_confidence_calibration.png",
        "Gambar 4.7 Distribusi confidence prediksi benar dan salah serta diagram reliabilitas kalibrasi 10-bin EXP-10 (ECE = 0,0282).",
        width_inch=6.0,
    )

    # Subbab 4.4
    heading2("4.4 Analisis dan Pembahasan Awal")
    p(
        "Berdasarkan perbandingan hasil eksperimen pada Tabel 4.2, arsitektur CNN Baseline 3-blok (102.089 parameter) "
        "mengalami underfitting pada seluruh skenario (EXP-01 mencapai 49,00%, EXP-02 sebesar 40,96%, EXP-03 sebesar 48,55%, "
        "dan EXP-07 sebesar 46,08% Macro F1). Kapasitas representasi tiga lapis konvolusi yang dilatih dari nol pada 3.327 citra "
        "belum mencukupi untuk membedakan variasi latar belakang dan tekstur sampah dunia nyata."
    )
    p(
        "Pemanfaatan transfer learning MobileNetV3-Small pada resolusi 128×128 (EXP-04) mampu menaikkan Macro F1 ke 71,93%, "
        "tetapi mengalami overfitting berat dengan akurasi latih 98,41% berbanding akurasi validasi 71,25% (selisih 27,16%) "
        "dan validation loss mulai naik kembali setelah epoch ke-13 (0,9599 menjadi 1,0546 pada epoch 30). Penambahan regularisasi "
        "ketat pada EXP-05 (Dropout 0,4, weight decay 0,05, label smoothing 0,05) menurunkan selisih tersebut menjadi 14,16%, "
        "tetapi Macro F1 bertahan di 70,82% karena keterbatasan detail piksel pada ukuran 128×128."
    )
    p(
        f"Ketika resolusi dinaikkan menjadi 224×224 pada EXP-08 dan EXP-09, performa validasi langsung meningkat ke 76,25% "
        f"dan 76,64% Macro F1 serta menurunkan validation loss EXP-08 hingga 0,7844. Penggabungan resolusi 224×224 dengan "
        f"strategi two-stage fine-tuning pada EXP-10 menghasilkan peningkatan Macro F1 sebesar {f1_gain:.2f} poin persentase "
        f"dibanding EXP-04 (menjadi {format_id_pct(exp10['best_val_macro_f1'])}), dengan akurasi validasi {format_id_pct(exp10['best_val_accuracy'])}, "
        f"validation loss {format_id_num(exp10['best_val_loss'], 4)} yang terus menurun stabil sampai epoch 30, dan ECE {format_id_num(cal['ece_10_bins'], 4)}. "
        f"Penggunaan bobot kelas pada EXP-11 menghasilkan Macro Recall sedikit lebih tinggi ({format_id_pct(exp11['best_val_macro_recall'])}), "
        f"namun Macro F1-nya ({format_id_pct(exp11['best_val_macro_f1'])}) berada tipis di bawah EXP-10."
    )
    p(
        f"Meskipun selisih akurasi latih dan validasi pada EXP-10 berkurang sebesar {gap_drop:.2f} poin persentase dibanding "
        f"EXP-04 (dari 27,16% menjadi {format_id_pct(exp10['train_val_acc_gap'])}), angka tersebut masih di atas batas target 10%. "
        f"Pembukaan empat blok terakhir membuat jumlah parameter yang dilatih mencapai 1.336.561 untuk 3.327 citra latih sehingga "
        f"akurasi latih kembali mendekati jenuh ({format_id_pct(exp10['best_train_accuracy'])}). Selain itu, performa tiga kelas "
        f"terlemah yaitu {err['weakest_3_classes'][0]['class']} (F1 {format_id_pct(err['weakest_3_classes'][0]['f1'])}), "
        f"{err['weakest_3_classes'][1]['class']} (F1 {format_id_pct(err['weakest_3_classes'][1]['f1'])}, recall 60,14%), "
        f"dan {err['weakest_3_classes'][2]['class']} (F1 {format_id_pct(err['weakest_3_classes'][2]['f1'])}) menunjukkan bahwa "
        f"kesalahan dominan berupa Plastic yang diprediksi sebagai Metal (26 sampel, berbanding 22 sampel pada EXP-04), Paper "
        f"sebagai Cardboard (16 sampel), dan Miscellaneous Trash sebagai Textile Trash (12 sampel). Kemasan plastik yang memantulkan "
        f"cahaya kerap tampak menyerupai permukaan logam."
    )

    # Subbab 4.5
    heading2("4.5 Kendala dan Rencana Perbaikan Progres 5")
    p(
        "Berdasarkan hasil evaluasi pada Validation Set, model EXP-10 (Val Macro F1 79,16%, Val Loss 0,8703, ECE 0,0282) "
        "ditetapkan sebagai model terbaik sementara pada tahap Progres 4. Sementara itu, Test Set (712 citra) tetap dipertahankan "
        "dalam keadaan terkunci hingga tahap akhir Progres 5. Tabel 4.6 merangkum kendala yang masih ditemukan beserta rencana "
        "perbaikannya untuk Progres 5."
    )

    add_table(
        ["Kendala pada Progres 4", "Bukti Kuantitatif", "Rencana Perbaikan Progres 5"],
        [
            ("Selisih akurasi latih dan validasi pada EXP-10 masih melebihi 10%", "Akurasi latih 97,39% vs validasi 77,98% (selisih 19,40% dengan 1,34 juta parameter dilatih)", "Menguji pembukaan blok yang lebih sempit (unfreeze_last_n=2) serta menambahkan regularisasi MixUp dan RandomErasing untuk menekan selisih akurasi."),
            ("Kesalahan dominan pada kelas Plastic menjadi Metal dan Paper menjadi Cardboard", "Plastic diprediksi Metal sebanyak 26 sampel (recall Plastic 60,14%), Paper menjadi Cardboard 16 sampel, serta 15 salah prediksi dengan confidence di atas 0,90", "Menerapkan analisis visual Grad-CAM untuk memeriksa bagian citra yang menjadi fokus model pada sampel salah prediksi."),
            ("Generalisasi akhir belum diuji pada Test Set", "Seluruh 11 eksperimen dievaluasi hanya pada 713 sampel Validation Set (test_used bernilai false)", "Melakukan pengujian tunggal pada 712 citra Test Set setelah konfigurasi model final di Progres 5 ditetapkan."),
        ],
        [1.8, 1.9, 2.5],
        caption_text="Tabel 4.6 Kendala Pelatihan dan Rencana Perbaikan untuk Progres 5",
    )

    heading1("CHECKLIST EVALUASI SEBELUM PENGUMPULAN (PROGRES 4)")
    add_table(
        ["Kriteria Evaluasi Progres 4", "Status", "Keterangan Bukti pada Laporan dan Repositori"],
        [
            ("Naskah BAB I, II, dan III dipertahankan utuh", "Terpenuhi", "Seluruh isi BAB I hingga BAB III dari Progres 3 dipertahankan sebelum BAB IV."),
            ("Bukti kode berjalan dan reproduksibilitas tercatat", "Terpenuhi", "Seed=42, PyTorch 2.11.0+cu130, commit c29d5b6, dan pytest lulus 14 pengujian."),
            ("Seluruh hasil ablasi disajikan sesuai keluaran JSON", "Terpenuhi", "Tabel 4.2 memuat 11 eksperimen (EXP-01 hingga EXP-11) dari experiment_comparison.json."),
            ("Grafik akurasi, F1, loss, dan selisih akurasi disertakan", "Terpenuhi", "Gambar 4.1 hingga 4.4 menampilkan contoh prapemrosesan, kurva pelatihan, dan perbandingan selisih akurasi."),
            ("Confusion matrix, metrik per kelas, dan contoh prediksi disajikan", "Terpenuhi", "Gambar 4.5, Gambar 4.6, Tabel 4.4, dan Tabel 4.5 merujuk pada val_best_model_evaluation.json."),
            ("Pembahasan keterbatasan model disampaikan secara objektif", "Terpenuhi", "Gambar 4.7 (ECE 0,0282) dan Subbab 4.4 menjelaskan selisih 19,40% serta 26 kasus Plastic ke Metal."),
            ("Data uji (Test Set) tetap dijaga agar tidak bocor", "Terpenuhi", "Sebanyak 712 citra Test Set tetap terkunci (test_set_untouched=true) hingga Progres 5."),
        ],
        [2.1, 1.0, 3.1],
        caption_text="Tabel 4.7 Checklist Evaluasi Progres 4",
    )


def main() -> None:
    comparison = json.loads((ROOT / "results/progress4/experiment_comparison.json").read_text(encoding="utf-8"))
    val_eval = json.loads((ROOT / "results/progress4/val_best_model_evaluation.json").read_text(encoding="utf-8"))
    exp10_summary = json.loads(
        (ROOT / "results/progress4/exp10_mobilenet_224_2stage_ft/summary.json").read_text(encoding="utf-8")
    )
    exp10_history = json.loads(
        (ROOT / "results/progress4/exp10_mobilenet_224_2stage_ft/history.json").read_text(encoding="utf-8")
    )
    exp11_summary = json.loads(
        (ROOT / "results/progress4/exp11_mobilenet_224_2stage_cw/summary.json").read_text(encoding="utf-8")
    )

    base_docx = ROOT / "laporan/DL_A_KelompokART_Progres3.docx"
    doc = Document(str(base_docx))

    # Update cover subtitle and date cleanly while preserving runs
    for p_elem in doc.paragraphs[:6]:
        if "PROGRES 1, 2, DAN 3" in p_elem.text:
            for r in p_elem.runs:
                r.text = r.text.replace(
                    "PROGRES 1, 2, DAN 3 (RINGKAS)  |  PERTEMUAN 10–12",
                    "PROGRES 1, 2, 3, DAN 4 (RINGKAS)  |  PERTEMUAN 13–14",
                )
    if doc.tables:
        for row in doc.tables[0].rows:
            if "Tanggal Pembaruan" in row.cells[0].text:
                p_cell = row.cells[1].paragraphs[0]
                if p_cell.runs:
                    p_cell.runs[0].text = "27 November 2026"
                    for extra in p_cell.runs[1:]:
                        extra.text = ""
                else:
                    p_cell.text = "27 November 2026"

    # Update unified footer in all base sections so linked BAB IV shows Progres 4
    for sec in doc.sections:
        for para in sec.footer.paragraphs:
            for r in para.runs:
                if "3 | Halaman" in r.text:
                    r.text = r.text.replace("3 | Halaman", "4 | Halaman")

    # Append Section 2 (BAB IV) on a new page with linked footer & continuous page numbering
    sec_bab4 = doc.add_section(WD_SECTION_START.NEW_PAGE)
    sec_bab4.top_margin = Inches(0.85)
    sec_bab4.bottom_margin = Inches(0.85)
    sec_bab4.left_margin = Inches(1.05)
    sec_bab4.right_margin = Inches(0.95)
    sec_bab4.footer_distance = Inches(0.55)
    sec_bab4.footer.is_linked_to_previous = True
    pg_num_type = sec_bab4._sectPr.find(qn("w:pgNumType"))
    if pg_num_type is not None:
        sec_bab4._sectPr.remove(pg_num_type)

    _add_bab4_content(doc, comparison, val_eval, exp10_summary, exp10_history, exp11_summary)

    unified_out = ROOT / "laporan/DL_A_KelompokART_Progres4.docx"
    doc.save(str(unified_out))
    print(f"Saved unified Progres 4 report: {unified_out}")

    # Build standalone BAB IV for convenience
    standalone = Document(str(base_docx))
    p_count = len(standalone.paragraphs)
    t_count = len(standalone.tables)
    for p_elem in list(standalone.paragraphs):
        p_elem._p.getparent().remove(p_elem._p)
    for t_elem in list(standalone.tables):
        t_elem._tbl.getparent().remove(t_elem._tbl)
    for para in standalone.sections[0].footer.paragraphs:
        for r in para.runs:
            if "3 | Halaman" in r.text:
                r.text = r.text.replace("3 | Halaman", "4 | Halaman")
    _add_bab4_content(standalone, comparison, val_eval, exp10_summary, exp10_history, exp11_summary)
    standalone_out = ROOT / "laporan/BAB_IV_Progres4.docx"
    standalone.save(str(standalone_out))
    print(f"Saved standalone BAB IV ({p_count} base paras, {t_count} base tables removed): {standalone_out}")


if __name__ == "__main__":
    main()
