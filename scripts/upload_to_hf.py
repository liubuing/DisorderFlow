#!/usr/bin/env python3
"""Upload all DisorderFlow datasets to Hugging Face.

Usage:
    python scripts/upload_to_hf.py

Uploads data/ directory to liubuing/disorderflow (dataset repo).
Large files (>10MB) are automatically tracked with Git LFS.
Supports resume if interrupted.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# 强制 UTF-8 运行环境，避免 Windows 控制台编码问题影响上传日志与文件名。
os.environ.setdefault("PYTHONUTF8", "1")

from huggingface_hub import HfApi

# 上传目标：HF 数据集仓库与本地 data/ 目录。
REPO_ID = "liubuing/disorderflow"
REPO_TYPE = "dataset"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# File patterns to track with LFS (large binary/data files)
# 需要 Git LFS 跟踪的大文件类型：LMDB/权重/数组/序列化对象/压缩包/
# 结构文件等二进制或大体积数据。
LFS_PATTERNS = [
    "*.mdb",
    "*.lmdb",
    "*.pt",
    "*.pth",
    "*.npy",
    "*.npz",
    "*.pkl",
    "*.tar.gz",
    "*.zip",
    "*.h5",
    "*.hdf5",
    "*.pdb",
    "*.cif",
    "*.tsv",
    "*.json",
]


def main():
    api = HfApi()

    # Verify auth
    # 第一步：确认 HF 凭证有效，未登录则提示并退出。
    try:
        user = api.whoami()["name"]
        print(f"[auth] Logged in as: {user}")
    except Exception as e:
        print(f"[error] Not authenticated: {e}")
        print("Run: hf auth login")
        sys.exit(1)

    # Verify repo exists
    # 第二步：目标仓库不存在时自动创建（公开数据集仓库）。
    try:
        api.repo_info(REPO_ID, repo_type=REPO_TYPE)
        print(f"[repo] Target: {REPO_ID} ({REPO_TYPE})")
    except Exception:
        print(f"[repo] Creating {REPO_ID} ...")
        api.create_repo(REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)

    # Set LFS tracking patterns
    # 第三步：预登记 LFS 跟踪规则；已设置过的模式会抛异常，直接忽略。
    print("[lfs] Setting LFS patterns...")
    for pattern in LFS_PATTERNS:
        try:
            api.preupload_lfs_files(REPO_ID, [pattern], repo_type=REPO_TYPE)
        except Exception:
            pass  # Pattern may already be set

    # Upload using upload_large_folder (designed for large datasets)
    # 第四步：用 upload_large_folder 整目录上传（自动分批 + 断点续传）。
    # data/ 体积约 220GB，中断后重新运行即可从已完成处继续。
    print(f"[upload] Source: {DATA_DIR}")
    print(f"[upload] This will take a long time for ~220GB of data.")
    print(f"[upload] Using upload_large_folder with auto-resume and batching.\n")

    start = time.time()
    try:
        api.upload_large_folder(
            folder_path=str(DATA_DIR),
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[error] {e}")
        print(f"[info] Elapsed: {elapsed/60:.1f} min. Re-run to resume from where it stopped.")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n[done] Upload complete in {elapsed/60:.1f} min")
    print(f"[done] View at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
