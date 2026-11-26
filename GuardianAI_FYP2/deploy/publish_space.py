"""
Publish the FYP-2 online demo to a Hugging Face Space
=====================================================
Copies the detector code and weights next to deploy/hf_space/app.py and
uploads them. Needs a valid login first:  .venv\\Scripts\\hf auth login

    .venv\\Scripts\\python deploy\\publish_space.py --stage-only     # build deploy/build/space only
    .venv\\Scripts\\python deploy\\publish_space.py                  # build + upload (public Space)
    .venv\\Scripts\\python deploy\\publish_space.py --name my-space
"""

import argparse
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SPACE_SRC = PROJECT / "deploy" / "hf_space"
STAGE = PROJECT / "deploy" / "build" / "space"
FILES = [
    "ai/__init__.py", "ai/contracts.py", "ai/adapters.py", "ai/fallnet.py",
    "ai/pose_features.py", "ai/tracking.py",
    "fall_detector_model.py", "fyp1_config.py", "live_demo.py",
    "weights/fallnet_v2.pt", "weights/yolov8n-pose.pt",
]


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(SPACE_SRC, STAGE)
    for rel in FILES:
        (STAGE / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / rel, STAGE / rel)
    print(f"Staged {len(list(STAGE.rglob('*')))} entries in {STAGE}")


def publish(name):
    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=repo_id, repo_type="space",
                      commit_message="GuardianAI FYP-2 demo")
    print(f"Uploaded. Space: https://huggingface.co/spaces/{repo_id}")
    print(f"App URL:  https://{user.lower()}-{name.lower()}.hf.space")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="guardianai-fyp2")
    p.add_argument("--stage-only", action="store_true")
    args = p.parse_args()
    stage()
    if not args.stage_only:
        publish(args.name)
