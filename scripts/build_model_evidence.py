"""Rebuild architecture figures and execute the dataset-free design notebook."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def diagrams() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams["svg.fonttype"] = "none"  # preserve editable text in SVG

    # Stage dimensions are obtained from actual modules using forward hooks.
    import torch
    from src.models import CNNBaseline, MobileNetV3Small
    torch.set_num_threads(2)
    baseline, mobile = CNNBaseline().eval(), MobileNetV3Small().eval()
    shapes = {}
    hooks = []
    stages = {"conv1": baseline.features[2], "conv2": baseline.features[5],
              "conv3": baseline.features[8], "avg": baseline.classifier[0],
              "features": mobile.network.features, "mobile_avg": mobile.network.avgpool}
    for key, module in stages.items():
        def capture(_module, _inputs, output, key=key):
            shapes[key] = list(output.shape)
        hooks.append(module.register_forward_hook(capture))
    with torch.inference_mode():
        baseline(torch.zeros(2, 3, 128, 128))
        mobile(torch.zeros(2, 3, 128, 128))
    for hook in hooks:
        hook.remove()
    def fmt(key):
        return "[B, " + ", ".join(map(str, shapes[key][1:])) + "]"
    specs = [
        ("cnn_baseline_architecture", "CNN baseline dari awal", [
            ("Input RGB", "[B, 3, 128, 128] | piksel [0,1]"),
            ("Blok 1", "Conv 3→32, 3×3, p=1 | ReLU | MaxPool 2×2\n" + fmt("conv1")),
            ("Blok 2", "Conv 32→64, 3×3, p=1 | ReLU | MaxPool 2×2\n" + fmt("conv2")),
            ("Blok 3", "Conv 64→128, 3×3, p=1 | ReLU | MaxPool 2×2\n" + fmt("conv3")),
            ("Feature pooling", "AdaptiveAvgPool 1×1 → Flatten | [B, 128]"),
            ("Classification head", "Linear 128→64 | ReLU | Dropout p=0,3"),
            ("Keluaran model", "Linear 64→9 | logits [B, 9]"),
            ("Konversi saat inferensi", "Softmax(dim=1) | probabilitas [B, 9]"),
        ]),
        ("mobilenetv3_small_architecture", "MobileNetV3-Small transfer learning", [
            ("Input RGB", "[B, 3, 128, 128] | normalisasi mean–std ImageNet"),
            ("Feature extractor torchvision (frozen)", "Conv awal + 11 inverted residual + Conv akhir\n13 blok features | " + fmt("features")),
            ("Feature pooling", "AdaptiveAvgPool 1×1 → Flatten | [B, 576]"),
            ("Classification head", "Linear 576→1024 | Hardswish | Dropout p=0,2"),
            ("Keluaran model", "Linear pengganti 1024→9 | logits [B, 9]"),
            ("Konversi saat inferensi", "Softmax(dim=1) | probabilitas [B, 9]"),
        ]),
    ]
    target = ROOT / "figures"
    for filename, title, rows in specs:
        fig, ax = plt.subplots(figsize=(8.1, len(rows) * 0.86 + 0.9))
        ax.set(xlim=(0, 10), ylim=(0, len(rows) * 1.25 + 0.8))
        ax.axis("off")
        ax.text(5, len(rows) * 1.25 + 0.4, title, ha="center", fontsize=15, weight="bold")
        for i, (heading, body) in enumerate(rows):
            y = (len(rows) - 1 - i) * 1.25 + 0.1
            color = "#ecf4fb" if i != len(rows)-1 else "#fff4de"
            ax.add_patch(FancyBboxPatch((0.25, y), 9.5, 0.93, boxstyle="round,pad=0.04", linewidth=1,
                                       edgecolor="#345878", facecolor=color))
            ax.text(5, y + 0.67, heading, ha="center", va="center", fontsize=11, weight="bold")
            ax.text(5, y + 0.29, body, ha="center", va="center", fontsize=10)
            if i < len(rows)-1:
                ax.annotate("", xy=(5, y-0.26), xytext=(5, y-0.03),
                            arrowprops={"arrowstyle": "->", "color": "#345878", "lw": 1.4})
        fig.savefig(target / f"{filename}.png", dpi=200, bbox_inches="tight")
        fig.savefig(target / f"{filename}.svg", bbox_inches="tight")
        plt.close(fig)


def notebook(execute: bool = True) -> None:
    import nbformat as nbf
    from nbclient import NotebookClient
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("# RealWaste — Progres 3: Perancangan dan Implementasi Awal Model\n\n"
           "Kelompok ART, Deep Learning Kelas A. Tujuan: menyiapkan perbandingan CNN baseline, "
           "augmentasi, pembobotan kelas, dan transfer learning MobileNetV3-Small. "
           "Notebook ini menguji arsitektur dengan tensor sintetis pada CPU. Tidak ada pelatihan atau evaluasi dataset."),
        md("## Kesiapan data dan spesifikasi\n\n"
           "Bukti historis Progres 2 mencatat 4.752 citra RGB 524 × 524 dan sembilan kelas. "
           "Data mentah tidak disertakan di repositori. Audit citra rusak dan duplikat identik selesai; "
           "tinjauan kandidat near-duplicate belum lengkap. Resize, cache, augmentasi, dan split belum dilakukan. "
           "Rencana input 128 × 128, tensor NCHW `[B,3,128,128]`; output logits `[B,9]`; "
           "satu label integer 0–8 per citra."),
        code("from pathlib import Path\nimport sys\nimport json\nimport torch\n"
             "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
             "sys.path.insert(0, str(ROOT))\n"
             "from src.models import CNNBaseline, MobileNetV3Small, CLASS_TO_IDX, load_config\n"
             "from src.verify_models import verify_models, build_initial_components\n"
             "torch.set_num_threads(2)\ntorch.manual_seed(42)\n"
             "config = load_config()\nprint(CLASS_TO_IDX)\nprint('PyTorch', torch.__version__, '| device: cpu')"),
        md("## CNN baseline\n\n"
           "Tiga blok Conv 3×3 + ReLU + MaxPool 2×2 dengan kanal 32, 64, 128. "
           "Global average pooling membatasi ukuran head; Linear 128→64, ReLU, dropout 0,3, "
           "dan Linear 64→9 menghasilkan logits. Piksel direncanakan diskalakan ke [0,1]. "
           "Model dari awal menjadi acuan tanpa pengetahuan ImageNet."),
        code("baseline = CNNBaseline(dropout=config['models']['cnn_baseline']['dropout']).eval()\nprint(baseline)"),
        md("## MobileNetV3-Small\n\n"
           "Backbone resmi torchvision, adaptive average pooling, Linear 576→1024, Hardswish, "
           "dropout 0,2, Linear pengganti 1024→9. Backbone awal dibekukan termasuk statistik "
           "BatchNorm; head tetap trainable. Rencana Progres 4 menggunakan IMAGENET1K_V1, "
           "lalu mempertimbangkan fine-tuning blok terakhir menggunakan validasi. "
           "Verifikasi berikut memakai `pretrained=False`, sehingga bobot ImageNet tidak diunduh/dimuat. "
           "Normalisasi rencana: mean [0,485;0,456;0,406], std [0,229;0,224;0,225] setelah skala [0,1]. "
           "Resize 128 merupakan adaptasi proposal; transform bawaan bobot menggunakan crop 224. "
           "Pengaruh resolusi perlu ditinjau pada tahap berikutnya."),
        code("mobile = MobileNetV3Small(pretrained=False, freeze_backbone=True, seed=42).eval()\n"
             "print(mobile)\nprint('ImageNet loaded:', mobile.pretrained_loaded)\n"
             "print('Frozen:', all(not p.requires_grad for p in mobile.network.features.parameters()))"),
        md("## Verifikasi forward sintetis\n\n"
           "Tensor acak adalah alat uji bentuk, bukan citra atau prediksi RealWaste. "
           "Policy normalisasi di atas belum menjadi pipeline data. Model berjalan dalam eval/inference mode."),
        code("dummy = torch.rand(2, 3, 128, 128)\n"
             "with torch.inference_mode():\n"
             "    for name, model in [('CNN', baseline), ('MobileNetV3', mobile)]:\n"
             "        logits = model(dummy)\n"
             "        assert logits.shape == (2, 9) and torch.isfinite(logits).all()\n"
             "        assert torch.equal(logits, model(dummy))\n"
             "        probabilities = logits.softmax(dim=1)\n"
             "        assert torch.allclose(probabilities.sum(1), torch.ones(2))\n"
             "        print(name, 'output:', list(logits.shape), 'finite:', bool(torch.isfinite(logits).all()))"),
        code("verification = verify_models(output_dir=None)\n"
             "for name, summary in verification['models'].items():\n"
             "    print(name, {k:v for k,v in summary.items() if k != 'architecture'})\n"
             "print(verification['versions'])"),
        md("## Konfigurasi awal dan perbandingan\n\n"
           "| Aspek | CNN baseline | MobileNetV3-Small |\n|---|---|---|\n"
           "| Peran | Pembanding dari awal | Arsitektur transfer learning utama |\n"
           "| Optimizer | Adam, lr 0,001 | AdamW, lr 0,0001 |\n"
           "| Batch size rencana | 32 | 16 |\n| Epoch maksimum rencana | 30 | 30 |\n"
           "| Bobot awal pengujian | Acak | Acak, backbone frozen |\n"
           "| Bobot rencana Progres 4 | Acak | ImageNet |\n\n"
           "CrossEntropyLoss dengan target integer merupakan padanan objektif sparse categorical crossentropy "
           "pada BAB II, dan menerima logits langsung. Softmax hanya ketika probabilitas dibutuhkan. "
           "Learning rate lebih kecil dan batch size lebih kecil disiapkan untuk transfer learning. "
           "AdamW memakai weight decay 0,01; CNN memakai dropout untuk regularisasi. Nilai ini usulan, belum optimal."),
        code("for model, name in [(baseline, 'cnn_baseline'), (mobile, 'mobilenet_v3_small')]:\n"
             "    loss, optimizer = build_initial_components(model, name)\n"
             "    print(name, type(loss).__name__, type(optimizer).__name__, optimizer.defaults)\n"
             "# Tidak ada backward(), optimizer.step(), atau loop epoch."),
        md("## Diagram arsitektur\n\nSumber SVG dapat diedit; generator di scripts/build_model_evidence.py. "
           "Kotak softmax adalah konversi inferensi di luar model."),
        code("from IPython.display import Image, display\n"
             "for filename in ['cnn_baseline_architecture.png', 'mobilenetv3_small_architecture.png']:\n"
             "    display(Image(filename=str(ROOT / 'figures' / filename), width=680))"),
        md("## Rencana eksperimen dan batasan\n\n"
           "Progres 4 memerlukan audit kandidat near-duplicate, split stratified group-aware 70/15/15 "
           "(3.326/713/713 hanya perkiraan; jumlah aktual mengikuti grup), dan pipeline input yang konsisten. "
           "Augmentasi hanya pada training; pembobotan dihitung dari training. Bandingkan kontrol tanpa/dengan "
           "augmentasi serta tanpa/dengan pembobotan, bukan mengaktifkan semuanya otomatis. "
           "Pantau loss dan akurasi serta precision/recall/F1 makro pada validasi; "
           "confusion matrix, selisih latih–validasi, waktu pelatihan dan Grad-CAM direncanakan kemudian. "
           "Testing hanya sekali setelah konfigurasi akhir.\n\n"
           "Inisialisasi dan bentuk tensor terverifikasi; hasil ini tidak membuktikan kemampuan klasifikasi. "
           "Belum ada pelatihan, tuning, evaluasi dataset atau bobot pretrained yang dimuat. "
           "GPU tidak digunakan dan kapasitas pelatihan pada CPU 8 GB belum dibuktikan.\n\n"
           "Referensi: [UCI RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste), "
           "[torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html), "
           "[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)."),
    ]
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3 (RealWaste)"}})
    path = ROOT / "notebooks/02_model_design.ipynb"
    if execute:
        NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, path)
    print(f"Notebook saved: {path}; executed={execute}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook-only", action="store_true")
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    if not args.notebook_only:
        diagrams()
    notebook(execute=not args.no_execute)
