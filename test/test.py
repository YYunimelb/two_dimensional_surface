import os
import shutil


def copy_missing_files(src_dir, check_dir, dest_dir):
    """
    Copy files from src_dir to dest_dir if they are not found in check_dir.

    :param src_dir: Path to the source directory containing files to check.
    :param check_dir: Path to the directory to check for the files.
    :param dest_dir: Path to the destination directory to copy missing files.
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # List all files in the source directory
    src_files = os.listdir(src_dir)

    for file_name in src_files:
        src_file_path = os.path.join(src_dir, file_name)

        # Skip directories, only process files
        if not os.path.isfile(src_file_path):
            continue

        # Check if the file exists in the check_dir
        check_file_path = os.path.join(check_dir, file_name)
        if not os.path.exists(check_file_path):
            # File does not exist in check_dir, copy to dest_dir
            dest_file_path = os.path.join(dest_dir, file_name)
            shutil.copy2(src_file_path, dest_file_path)
            print(f"Copied: {file_name} -> {dest_dir}")


# Example usage
if __name__ == "__main__":
    # Paths to your directories
    src_directory = "../data/structure_not_satisfy"
    check_directory = "../data/check_layer_connectivity"
    dest_directory = "../data/structure_not_satisfy2"

    copy_missing_files(src_directory, check_directory, dest_directory)
