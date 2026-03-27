import os
import shutil
from pathlib import Path

COMMON_ENV_NAMES = {
    ".venv",
    "venv",
    ".conda",
    "env",
}

ENV_PREFIXES = (
    "venv-",
    ".venv-",
    "conda-",
    ".conda-",
)


def is_probable_env_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    name = path.name.lower()

    # Rule 1: common exact env names
    if name in COMMON_ENV_NAMES:
        return True

    # Rule 2: common prefixes like venv-myname
    if any(name.startswith(prefix) for prefix in ENV_PREFIXES):
        return True

    # Rule 3: real venv markers inside the folder itself
    pyvenv_cfg = path / "pyvenv.cfg"
    scripts_python = path / "Scripts" / "python.exe"
    bin_python = path / "bin" / "python"
    conda_meta = path / "conda-meta"

    if pyvenv_cfg.exists():
        return True
    if scripts_python.exists():
        return True
    if bin_python.exists():
        return True
    if conda_meta.exists():
        return True

    return False


def find_env_dirs(root: Path) -> list[Path]:
    found = []

    for current_root, dirnames, _ in os.walk(root):
        current_path = Path(current_root)

        # iterate over child directories only
        for dirname in list(dirnames):
            candidate = current_path / dirname

            if is_probable_env_dir(candidate):
                found.append(candidate)

        # don't walk down into detected envs
        dirnames[:] = [
            d for d in dirnames
            if not is_probable_env_dir(current_path / d)
        ]

    # unique paths
    unique = []
    seen = set()
    for p in found:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


def remove_dirs(dirs: list[Path]) -> None:
    if not dirs:
        print("\nNo environment folders found.")
        return

    print("\nEnvironment folders to delete:")
    for d in dirs:
        print(f"  - {d}")

    confirm = input("\nDelete these folders? (yes/no): ").strip().lower()
    if confirm not in {"yes", "y"}:
        print("Deletion cancelled.")
        return

    for d in dirs:
        try:
            shutil.rmtree(d)
            print(f"Deleted: {d}")
        except Exception as e:
            print(f"Failed to delete {d}: {e}")


def main():
    print("=== Python Environment Cleanup Tool ===")
    print("This script deletes virtual-environment folders only.\n")

    root_input = input("Enter the root folder to scan (default: current folder): ").strip()
    root = Path(root_input).expanduser().resolve() if root_input else Path.cwd().resolve()

    if not root.exists() or not root.is_dir():
        print(f"Invalid folder: {root}")
        return

    print(f"\nScanning for environment folders under:\n  {root}")
    env_dirs = find_env_dirs(root)
    remove_dirs(env_dirs)

    print("\nDone.")


if __name__ == "__main__":
    main()