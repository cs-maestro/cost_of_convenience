#!/usr/bin/env python3
import os
import hashlib
import shutil

# CONFIGURATION
INPUT_DIR     = "../../urls_raw_data/ss_data"
OUTPUT_DIR    = "./unique_ss"
IMAGE_EXTS    = {".png"}

def compute_md5(path, chunk_size=4096):
    """
    Compute the MD5 checksum of a file in binary mode.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def gather_image_paths(root, exts=IMAGE_EXTS):
    """
    Recursively collect all file paths under `root` whose extension is in `exts`.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                paths.append(os.path.join(dirpath, fn))
    return paths


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Gather all images
    all_paths = gather_image_paths(INPUT_DIR)
    print(f"Found {len(all_paths)} images in '{INPUT_DIR}'")

    # 2) Deduplicate by MD5 checksum
    seen_hashes = {}
    deduped_paths = []
    for path in all_paths:
        file_hash = compute_md5(path)
        if file_hash not in seen_hashes:
            seen_hashes[file_hash] = path
            deduped_paths.append(path)
        else:
            print(f"[MD5 DUP] {path} ≡ {seen_hashes[file_hash]}")

    print(f"{len(deduped_paths)} unique screenshots remain after deduplication")

    # 3) Copy each unique screenshot and its paired HTML (if exists)
    for screenshot in deduped_paths:
        base_name = os.path.splitext(os.path.basename(screenshot))[0]

        # Destination paths
        dest_image = os.path.join(OUTPUT_DIR, base_name + '.png')

        # Copy screenshot
        shutil.copy2(screenshot, dest_image)

    print(f"All unique screenshots copied to '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
