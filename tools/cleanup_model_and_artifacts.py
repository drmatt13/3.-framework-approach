from pathlib import Path


def delete_files_in_directory(directory: Path) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0

    deleted_count = 0
    for path in directory.rglob("*"):
        if path.is_file() or path.is_symlink():
            path.unlink()
            deleted_count += 1

    return deleted_count


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    target_dirs = ["model", "models", "artifacts"]

    total_deleted = 0
    for folder in target_dirs:
        folder_path = root_dir / folder
        deleted = delete_files_in_directory(folder_path)
        if folder_path.exists() and folder_path.is_dir():
            print(f"Deleted {deleted} file(s) from: {folder_path}")
        else:
            print(f"Skipped (not found): {folder_path}")
        total_deleted += deleted

    print(f"Total files deleted: {total_deleted}")


if __name__ == "__main__":
    main()