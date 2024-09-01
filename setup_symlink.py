from pathlib import Path
import sys

def create_symlink():
    # Get the current directory where the script is located
    repo_dir = Path(__file__).parent.resolve()

    # Define the original path and the symlink path relative to the repo directory
    original_path = repo_dir / 'libs' / 'NLP-on-multilingual-coin-datasets'
    symlink_path = repo_dir / 'libs' / 'NLP_on_multilingual_coin_datasets'
    
    if not symlink_path.exists():
        try:
            symlink_path.symlink_to(original_path)
            print(f"Symlink created: {symlink_path} -> {original_path}")
        except OSError as e:
            print(f"Failed to create symlink: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Symlink already exists, no action taken.")

if __name__ == "__main__":
    create_symlink()
