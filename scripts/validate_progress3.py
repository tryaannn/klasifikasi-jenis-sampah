"""Append BAB III and prove historical PDF and repository evidence preservation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import nbformat
import pymupdf
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "laporan/DL_A_KelompokART_Progres2.pdf"
    chapter = ROOT / "laporan/BAB_III_Progres3.pdf"
    target = ROOT / "laporan/DL_A_KelompokART_Progres3.pdf"
    historical = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    preserved = {}
    for relative in historical:
        if relative in ("README.md", ".gitignore"):
            continue  # explicitly authorized documentation changes
        actual = (ROOT / relative).read_bytes()
        original_blob = subprocess.check_output(["git", "rev-parse", f"HEAD:{relative}"], cwd=ROOT).strip()
        actual_blob = subprocess.check_output(["git", "hash-object", "--path", relative, relative], cwd=ROOT).strip()
        # Respect pre-existing Git checkout CRLF filters for text files.
        assert actual_blob == original_blob, f"Historical file changed: {relative}"
        if relative.lower().endswith((".pdf", ".png")):
            original = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
            assert actual == original, f"Historical binary changed: {relative}"
        preserved[relative] = hashlib.sha256(actual).hexdigest()
    writer = PdfWriter()
    writer.append(PdfReader(source))
    writer.append(PdfReader(chapter))
    with target.open("wb") as stream:
        writer.write(stream)
    old = pymupdf.open(source)
    merged = pymupdf.open(target)
    chapter_doc = pymupdf.open(chapter)
    pages = []
    for i, original in enumerate(old):
        before = original.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        after = merged[i].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        assert before.samples == after.samples
        assert original.get_text() == merged[i].get_text()
        assert original.rect == merged[i].rect
        pages.append({"page": i+1, "raster_identical": True, "text_identical": True,
                      "raster_sha256": hashlib.sha256(before.samples).hexdigest()})
    qa = ROOT / "tmp/qa"
    qa.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(chapter_doc):
        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(qa / f"chapter-{i+1}.png")
        text = page.get_text()
        assert text.strip(), f"Empty chapter page {i+1}"
        assert f"Halaman {len(old)+i+1}" in text, f"Pagination mismatch: {i+1}"
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    assert x0 >= 25 and x1 <= page.rect.width-25 and y0 >= 20 and y1 <= page.rect.height-20, span
    full_text = "\n".join(p.get_text() for p in chapter_doc)
    for heading in ("3.1 Ringkasan", "3.2 Pemilihan", "3.3 Rancangan", "3.4 Konfigurasi", "3.5 Sampel"):
        assert heading in full_text
    assert "3.6" not in full_text
    nb = nbformat.read(ROOT / "notebooks/02_model_design.ipynb", as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    for excerpt in json.loads((ROOT / "results/model_design/report_code_excerpts.json").read_text(encoding="utf-8")):
        lines = (ROOT / excerpt["source"]).read_text(encoding="utf-8").splitlines()
        assert excerpt["text"] == "\n".join(lines[excerpt["start_line"]-1:excerpt["end_line"]])
    result = {
        "original_pdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "historical_files_unchanged_against_HEAD": preserved,
        "original_pages": len(old), "chapter_pages": len(chapter_doc), "merged_pages": len(merged),
        "preserved_pages": pages, "notebook_executed_code_cells": len(code_cells),
        "notebook_errors": 0, "chapter_pagination_valid": True,
        "chapter_text_inside_page_boundaries": True, "source_excerpts_match": True,
        "rendering": "LibreOffice 26.8.3 PDF export and render_docx.py PNG QA; PyMuPDF preservation checks",
    }
    (ROOT / "results/model_design/report_preservation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("historical_files_unchanged_against_HEAD", "preserved_pages")}, indent=2))


if __name__ == "__main__":
    main()
