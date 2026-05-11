#!/usr/bin/env python3
import os
import urllib.request
import json

API_URL = "https://api.github.com/repos/bacinger/f1-circuits/contents/circuits"
OUTPUT_DIR = "circuits"


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def download_file(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        with open(dest, "wb") as f:
            f.write(resp.read())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Fetching file list from {API_URL} ...")
    files = fetch_json(API_URL)

    total = len(files)
    for i, entry in enumerate(files, 1):
        name = entry["name"]
        dest = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(dest):
            print(f"[{i}/{total}] Skipping (already exists): {name}")
            continue
        print(f"[{i}/{total}] Downloading: {name}")
        download_file(entry["download_url"], dest)

    print(f"\nDone. {total} files in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
