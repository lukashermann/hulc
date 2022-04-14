import argparse
import multiprocessing
import os
from pathlib import Path
import subprocess

import numpy as np

from hulc.utils.utils import get_all_checkpoints


def get_log_dir(log_dir):
    log_dir = Path(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def intervals(parts, duration):
    part_duration = duration / parts
    return [str(int(i * part_duration)) + "-" + str(int(((i + 1) * part_duration) - 1)) for i in range(parts)]


def main():
    """
    This script calls the evaluate.sh script of the specified training_dir 8 times with different checkpoints
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument("--max_epoch", type=int, default=30, help="Evaluate until which epoch.")
    parser.add_argument("--log_dir", type=str, help="If calvin_agent was used to train, specify path to the log dir.")

    args = parser.parse_args()
    log_dir = get_log_dir(args.log_dir)

    eval_script = (Path(__file__).parent / "evaluate_policy.py").as_posix()
    training_dir = Path(args.train_folder)
    checkpoints = get_all_checkpoints(training_dir)
    epochs = [str(e) for chk in checkpoints if (e := int(chk.stem.split("=")[1])) <= args.max_epoch]
    split_epochs = np.array_split(epochs, 8)
    epoch_args = [",".join(arr) for arr in split_epochs]
    max_cpu_count = multiprocessing.cpu_count()
    local_cpus = intervals(8, max_cpu_count)
    for i, epoch_arg in enumerate(epoch_args):
        cmd = [
            "taskset",
            "--cpu-list",
            local_cpus[i],
            "python",
            eval_script,
            "--checkpoints",
            epoch_arg,
            "--dataset_path",
            args.dataset_path,
            "--train_folder",
            args.train_folder,
            "--log_dir",
            args.log_dir,
            "--device",
            str(i),
        ]
        std_out = log_dir / f"stdout_{i}.out"
        std_err = log_dir / f"stderr_{i}.err"
        with open(std_out, "wb") as out, open(std_err, "wb") as err:
            pro = subprocess.Popen(cmd, stdout=out, stderr=err, preexec_fn=os.setpgrp)


if __name__ == "__main__":
    main()
