"""Push the trained model to the HF Hub and the demo to HF Spaces.

Usage (after `hf auth login`):
  python scripts/push_to_hf.py --user <hf-username>            # model + space
  python scripts/push_to_hf.py --user <hf-username> --model-only

Model repo gets : best.pt, README.md (model card), metrics.json, confusion_matrix.png
Space repo gets : app.py, model.py, common.py, requirements.txt, README.md
                  (front-matter pins sdk: gradio), examples/*.wav
The Space loads weights from the model repo via TINYKWS_MODEL_REPO.
"""

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent


def ops_from(pairs):
    return [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src))
            for src, dst in pairs if Path(src).exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--model-only", action="store_true")
    ap.add_argument("--space-only", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    model_repo = f"{args.user}/tiny-kws"
    space_repo = f"{args.user}/tiny-kws"

    if not args.space_only:
        api.create_repo(model_repo, repo_type="model", exist_ok=True)
        ops = ops_from([
            (ROOT / "checkpoints/best.pt", "best.pt"),
            (ROOT / "app/MODEL_CARD.md", "README.md"),
            (ROOT / "assets/metrics.json", "metrics.json"),
            (ROOT / "assets/confusion_matrix.png", "confusion_matrix.png"),
        ])
        api.create_commit(model_repo, repo_type="model", operations=ops,
                          commit_message="model + card + eval artifacts")
        print(f"model:  https://huggingface.co/{model_repo}")

    if not args.model_only:
        api.create_repo(space_repo, repo_type="space", space_sdk="gradio",
                        exist_ok=True)
        pairs = [
            (ROOT / "app/app.py", "app.py"),
            (ROOT / "src/model.py", "model.py"),
            (ROOT / "src/common.py", "common.py"),
            (ROOT / "app/requirements.txt", "requirements.txt"),
            (ROOT / "app/README_space.md", "README.md"),
        ]
        pairs += [(p, f"examples/{p.name}")
                  for p in sorted((ROOT / "app/examples").glob("*.wav"))]
        api.create_commit(space_repo, repo_type="space",
                          operations=ops_from(pairs),
                          commit_message="demo app")
        api.add_space_variable(space_repo, "TINYKWS_MODEL_REPO", model_repo)
        print(f"space:  https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
