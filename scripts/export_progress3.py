"""Export BAB III with local/system LibreOffice; no system settings changes."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soffice", help="Optional path to LibreOffice executable")
    args = parser.parse_args()
    local = ROOT / ".cache/report-tools/libreoffice/program/soffice.com"
    executable = args.soffice or (str(local) if local.is_file() else shutil.which("soffice.com") or shutil.which("soffice"))
    if not executable:
        parser.error("LibreOffice unavailable. Install it or export the DOCX manually as laporan/BAB_III_Progres3.pdf")
    tmp = ROOT / "tmp"
    tmp.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lo-export-", dir=tmp) as profile:
        subprocess.run([
            executable, "--headless", f"-env:UserInstallation={Path(profile).as_uri()}",
            "--convert-to", "pdf:writer_pdf_Export", "--outdir", str(ROOT / "laporan"),
            str(ROOT / "laporan/BAB_III_Progres3.docx"),
        ], check=True, timeout=120)
    assert (ROOT / "laporan/BAB_III_Progres3.pdf").is_file()


if __name__ == "__main__":
    main()
