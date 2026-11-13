"""EDA RealWaste (UCI id 908) untuk BAB II laporan Progres 2.

Menghasilkan inventaris citra, statistik kelas/resolusi/kualitas, deteksi
duplikat (exact + near-duplicate), dan gambar pendukung. Ringan: hanya
Pillow, NumPy, pandas, matplotlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="EDA dataset RealWaste")
parser.add_argument("--raw", default=str(ROOT / "dataset" / "realwaste-main" / "RealWaste"),
                    help="folder induk yang berisi subfolder kelas")
parser.add_argument("--out", default=str(ROOT / "analysis"),
                    help="folder keluaran CSV, JSON, dan gambar")
ARGS = parser.parse_args()
RAW = Path(ARGS.raw)
OUT = Path(ARGS.out)
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


records = []
corrupt = []
files = sorted(p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
for p in files:
    rel = p.relative_to(RAW)
    cls = rel.parts[0]
    rec = {"file": str(rel).replace("\\", "/"), "kelas": cls,
           "ekstensi": p.suffix.lower().lstrip("."), "ukuran_byte": p.stat().st_size}
    try:
        with Image.open(p) as im:
            im.load()
            rec["lebar"], rec["tinggi"] = im.size
            rec["mode"] = im.mode
            rec["md5"] = hashlib.md5(p.read_bytes()).hexdigest()
            rec["dhash"] = dhash(im)
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            rec["brightness"] = float(arr.mean())
            rec["kontras"] = float(arr.std())
            rec["r"], rec["g"], rec["b"] = [float(arr[..., i].mean()) for i in range(3)]
    except Exception as exc:  # pragma: no cover
        corrupt.append({"file": str(rel), "error": str(exc)})
        continue
    records.append(rec)

df = pd.DataFrame(records).sort_values(["kelas", "file"]).reset_index(drop=True)
df.to_csv(OUT / "inventory.csv", index=False, encoding="utf-8")

# ---------- distribusi kelas ----------
dist = (df.groupby("kelas").size().rename("jumlah").reset_index()
        .sort_values("jumlah", ascending=False))
dist["persentase"] = (dist["jumlah"] / dist["jumlah"].sum() * 100).round(2)
dist["rasio_terhadap_min"] = (dist["jumlah"] / dist["jumlah"].min()).round(2)
dist.to_csv(OUT / "class_distribution.csv", index=False, encoding="utf-8")

# ---------- duplikat ----------
exact = df.groupby("md5")["file"].apply(list)
exact_dup = [g for g in exact if len(g) > 1]
near = []
dh = df["dhash"].to_numpy()
idx = df.index.to_numpy()
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if hamming(int(dh[i]), int(dh[j])) <= 6:
            near.append({"a": df.at[idx[i], "file"], "b": df.at[idx[j], "file"],
                         "kelas_a": df.at[idx[i], "kelas"], "kelas_b": df.at[idx[j], "kelas"],
                         "jarak": hamming(int(dh[i]), int(dh[j]))})
near_df = pd.DataFrame(near)
if not near_df.empty:
    near_df["beda_kelas"] = near_df["kelas_a"] != near_df["kelas_b"]
    near_df = near_df.sort_values("jarak").reset_index(drop=True)
near_df.to_csv(OUT / "near_duplicates.csv", index=False, encoding="utf-8")

# ---------- resolusi ----------
res = df.groupby(["lebar", "tinggi"]).size().rename("jumlah").reset_index()
res["rasio"] = (res["lebar"] / res["tinggi"]).round(3)
res.to_csv(OUT / "resolutions.csv", index=False, encoding="utf-8")

# ---------- ringkasan ----------
summary = {
    "total_citra": int(len(df)),
    "jumlah_kelas": int(df["kelas"].nunique()),
    "kelas": dist.to_dict("records"),
    "ekstensi": df["ekstensi"].value_counts().to_dict(),
    "mode_warna": df["mode"].value_counts().to_dict(),
    "resolusi_unik": int(len(res)),
    "resolusi": res.sort_values("jumlah", ascending=False).to_dict("records")[:5],
    "berkas_rusak": corrupt,
    "duplikat_identik_grup": len(exact_dup),
    "duplikat_identik_citra": int(sum(len(g) for g in exact_dup)),
    "near_duplicate_pasangan (dhash<=6)": int(len(near_df)),
    "near_duplicate_lintas_kelas": int(near_df["beda_kelas"].sum()) if not near_df.empty else 0,
    "ukuran_byte": {
        "min": int(df["ukuran_byte"].min()), "median": int(df["ukuran_byte"].median()),
        "max": int(df["ukuran_byte"].max()), "total_MB": round(df["ukuran_byte"].sum() / 1e6, 2),
    },
    "brightness": {
        "min": round(float(df["brightness"].min()), 2),
        "median": round(float(df["brightness"].median()), 2),
        "max": round(float(df["brightness"].max()), 2),
    },
    "brightness_per_kelas": {k: round(float(v), 2) for k, v in
                             df.groupby("kelas")["brightness"].median().items()},
    "kontras_per_kelas": {k: round(float(v), 2) for k, v in
                          df.groupby("kelas")["kontras"].median().items()},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

# ---------- grafik ----------
plt.rcParams.update({"font.size": 10, "figure.dpi": 200})
fig, ax = plt.subplots(figsize=(7.2, 3.6))
bars = ax.bar(dist["kelas"], dist["jumlah"], color="#4c72b0")
for b, n in zip(bars, dist["jumlah"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, str(n),
            ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Jumlah citra")
ax.set_title("Distribusi kelas RealWaste (n = 4.752)")
ax.set_ylim(0, dist["jumlah"].max() * 1.15)
plt.xticks(rotation=25, ha="right")
fig.tight_layout()
fig.savefig(FIG / "eda_01_distribusi_kelas.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
axes[0].hist(df["brightness"], bins=40, color="#55a868")
axes[0].set_title("Sebaran kecerahan rata-rata")
axes[0].set_xlabel("Nilai piksel rata-rata (0–255)")
axes[0].set_ylabel("Jumlah citra")
axes[1].hist(df["ukuran_byte"] / 1024, bins=40, color="#c44e52")
axes[1].set_title("Sebaran ukuran berkas")
axes[1].set_xlabel("Ukuran berkas (KB)")
axes[1].set_ylabel("Jumlah citra")
fig.tight_layout()
fig.savefig(FIG / "eda_02_kecerahan_ukuran.png")
plt.close(fig)

order = dist["kelas"].tolist()
data = [df.loc[df["kelas"] == k, "brightness"].to_numpy() for k in order]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
bp = ax.boxplot(data, labels=order, showfliers=False, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#9ecae1")
ax.set_ylabel("Kecerahan rata-rata")
ax.set_title("Variasi pencahayaan antarkelas")
plt.xticks(rotation=25, ha="right")
fig.tight_layout()
fig.savefig(FIG / "eda_03_kecerahan_per_kelas.png")
plt.close(fig)

# contoh 3 citra per kelas: 9 kolom (kelas) x 3 baris (sampel)
rng = np.random.default_rng(42)
n_show = 3
cell = 128
label_h = 30
sheet = Image.new("RGB", (len(order) * cell, n_show * cell + label_h), "white")
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 13)
except OSError:
    font = ImageFont.load_default()
chosen = {}
for c, cls in enumerate(order):
    sub = df[df["kelas"] == cls].sort_values("file")
    picks = rng.choice(sub["file"].to_numpy(), size=min(n_show, len(sub)), replace=False)
    chosen[cls] = picks.tolist()
    label = cls if len(cls) <= 14 else cls[:13] + "."
    draw.text((c * cell + 4, 8), label, fill="black", font=font)
    for r, rel in enumerate(picks):
        with Image.open(RAW / rel) as im:
            thumb = im.convert("RGB").resize((cell - 4, cell - 4), Image.LANCZOS)
        sheet.paste(thumb, (c * cell + 2, label_h + r * cell + 2))
sheet = sheet.resize((sheet.width * 2, sheet.height * 2), Image.LANCZOS)
sheet.save(FIG / "eda_04_contoh_per_kelas.png")
(OUT / "sample_per_kelas.json").write_text(
    json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps({k: v for k, v in summary.items()
                  if k not in ("kelas", "resolusi", "brightness_per_kelas",
                               "kontras_per_kelas")}, indent=2, ensure_ascii=False))
print(dist.to_string(index=False))
print(res.to_string(index=False))
