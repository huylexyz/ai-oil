"""Report project structure and integration handoff status."""

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "results",
    "reports",
    "slides",
    "docs",
    "scripts",
    "app.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
)
MODULES = ("data_loader.py", "features.py", "train.py", "predict.py", "evaluate.py")


def main() -> int:
    missing = [item for item in REQUIRED_PATHS if not (ROOT / item).exists()]
    if missing:
        print("Thiếu cấu trúc: " + ", ".join(missing))
        return 1

    print("OK: cấu trúc dự án và file cấu hình cơ bản đã có.")
    data_path = ROOT / "data" / "raw" / "DCOILBRENTEU_2015-2026.csv"
    print(f"Dữ liệu Brent: {'có' if data_path.is_file() else 'chưa có'} ({data_path})")

    for module in MODULES:
        path = ROOT / "src" / module
        state = "có khung; chờ bàn giao chức năng" if path.is_file() else "chưa có"
        if module == "data_loader.py" and path.is_file():
            state = "đã có loader dữ liệu lịch sử"
        print(f"src/{module}: {state}")

    artifacts = [
        item
        for item in (ROOT / "models").glob("*")
        if item.is_file() and item.name != ".gitkeep"
    ]
    print(f"Artifact mô hình: {len(artifacts)} tệp")
    print("Dự báo và đánh giá: chưa khả dụng nếu chưa bàn giao code/artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
