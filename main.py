import os
from pathlib import Path

from src.raptor_pipeline import RetrievalAugmentation


def load_env_file(filepath: str = ".env") -> None:
    """Manually load environment variables from .env if present."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key, val = parts[0].strip(), parts[1].strip()
                    # Only set if not already present in environment
                    if key not in os.environ:
                        os.environ[key] = val


def main() -> None:
    load_env_file()

    # Verify if HuggingFace token is configured (needed for gated models like LLaMA)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")

    if not hf_token or hf_token == "your_huggingface_token_here":
        print("[WARNING] No HF_TOKEN configured in .env. Gated model downloads (e.g., LLaMA) may fail.")


    # Configure paths
    project_root = Path(__file__).resolve().parent
    json_path = project_root / "data" / "sec_extracted_text" / "HAVA_extracted.json"
    output_dir = project_root / "data" / "trees" / "HAVA_extracted"

    if not json_path.exists():
        print(f"[ERROR] Sample file not found at: {json_path}")
        return

    print("Initializing RAPTOR with API-based models...")
    try:
        ra = RetrievalAugmentation()
    except Exception as e:
        print(f"[ERROR] Failed to initialize RetrievalAugmentation: {e}")
        return

    print(f"Building summary tree from: {json_path}...")
    try:
        tree = ra.add_json_document(json_path)
    except Exception as e:
        print(f"[ERROR] Failed to construct RAPTOR tree: {e}")
        return

    print("Tree constructed successfully!")
    print(f"  - Number of layers: {ra.num_layers}")
    print(f"  - Root nodes: {len(ra.root_summaries)}")

    print(f"Saving tree to: {output_dir}...")
    try:
        ra.save(output_dir)
        print("Done!")
    except Exception as e:
        print(f"[ERROR] Failed to save RAPTOR tree: {e}")


if __name__ == "__main__":
    main()


