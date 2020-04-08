"""This file creates a new data file like train.json in slurm_tmpdir
and zips the data there:
1. read data file list from file
2. create zip file of the data
3. copy it to slurm_tmpdir
4. create new data file list updated to use data on slurm_tmpdir
"""
import shutil
import zipfile
import argparse
import json
from pathlib import Path
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path to file containing the data list")
    parser.add_argument("-l", "--limit", help="Limit number of samples", default=-1)
    args = parser.parse_args()

    dir = Path(args.file).resolve().parent

    # load data list [{"x": p1, "m": p2}, ...]
    with open(args.file, "r") as f:
        data = json.load(f)

    if args.limit > 0:
        data = data[: args.limit]

    # create a list of files to zip, per domain ("x" and "m" for instance)
    zips = {}
    for d in data:
        for k, v in d.items():
            if k not in zips:
                zips[k] = []
            zips[k].append(v)

    print("Files listed")

    # create the actual zip file from the lists of paths
    for k, v in zips.items():
        zip_path = dir / f"{Path(args.file).stem}_{k}.zip"
        print("Creating", zip_path)
        if not zip_path.exists():
            with zipfile.ZipFile(zip_path, "w") as zipMe:
                for i, path in enumerate(v):
                    if i % 25 == 0:
                        print(i, end="\r", flush=True)
                    zipMe.write(
                        path,
                        Path(zip_path.stem) / Path(path).name,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
            print("Ok. Copying...", end="", flush=True)
        else:
            print("Using already existing zip. Copying...", end="", flush=True)
        # copy zip file to slurm_tmpdir
        shutil.copyfile(zip_path, Path(os.environ["SLURM_TMPDIR"]) / zip_path.name)
        # write new data list file to slurm_tmpdir
        with open(Path(os.environ["SLURM_TMPDIR"]) / Path(args.file).name, "w") as f:
            json.dump(
                [
                    {
                        k: str(
                            Path(os.environ["SLURM_TMPDIR"])
                            / Path(zip_path.stem)
                            / Path(v).name
                        )
                        for k, v in d.items()
                    }
                    for d in data
                ],
                f,
            )
        print("ok. Unzipping...", end="", flush=True)
        # unzip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(Path(os.environ["SLURM_TMPDIR"]))
