"""
Publish the browser workspace (deploy/static_space) as a free static Hugging Face Space.
    $env:HF_TOKEN = '<write token>';  .venv\\Scripts\\python deploy\\publish_static.py
The Django stylesheet (static/app.css) is copied in first so both UIs stay identical.
Test helpers (test_*.mjs, make_parity_fixture.py) are not uploaded; files from the
earlier FYP1-style demo are removed from the Space.
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "deploy" / "static_space"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="guardianai-fyp2")
    args = p.parse_args()
    shutil.copy2(PROJECT / "static" / "app.css", SRC / "app.css")
    api = HfApi()
    user = api.whoami()["name"]
    repo = f"{user}/{args.name}"
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        folder_path=str(SRC), repo_id=repo, repo_type="space", commit_message="GuardianAI FYP-2 browser workspace",
        ignore_patterns=["test_*.mjs", "make_parity_fixture.py", "__pycache__/*"],
        delete_patterns=["style.css", "fyp2.css"],
    )
    print(f"Space: https://huggingface.co/spaces/{repo}")
    print(f"Live:  https://{user.lower()}-{args.name}.static.hf.space/index.html")
