from pathlib import Path
import subprocess
import sys

from calvin_agent.utils.utils import get_all_checkpoints
import numpy as np


def main():
    """
    This script calls the evaluate.sh script of the specified training_dir 8 times with different checkpoints
    """
    training_dir = Path(sys.argv[1])
    eval_log_dir = training_dir / "evaluation"
    max_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else np.inf

    checkpoints = get_all_checkpoints(training_dir)
    epochs = [str(e) for chk in checkpoints if (e := int(chk.stem.split("=")[1])) <= max_epoch]
    split_epochs = np.array_split(epochs, 8)
    epoch_args = [",".join(arr) for arr in split_epochs if len(arr)]
    for epoch_arg in epoch_args:
        cmd = [(training_dir / "evaluate.sh").as_posix(), "--checkpoints", epoch_arg, "--eval_log_dir", eval_log_dir]
        output = subprocess.check_output(cmd)
        print(output.decode("utf-8"))


if __name__ == "__main__":
    main()
