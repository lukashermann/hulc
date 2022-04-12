from functools import partial
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

WAIT_INTERVAL = 60
LOG_DIR = Path(__file__).absolute().parents[2]


def error_file_name(i):
    path = LOG_DIR / f"stderr_{i}.err"
    return path.as_posix()


def out_file_name(i):
    path = LOG_DIR / f"stdout_{i}.out"
    return path.as_posix()


def check_dir_for_previous_files():
    files = sorted(LOG_DIR.glob("stderr_*"), reverse=True)
    print(files)
    if len(files):
        return int(files[0].stem.split("_")[1]) + 1
    return 0


def run_training(i, command):
    print("Start new training with command:")
    print(f"{' '.join(command)}")
    with open(out_file_name(i), "wb") as out, open(error_file_name(i), "wb") as err:
        pro = subprocess.Popen(command, stdout=out, stderr=err, preexec_fn=os.setpgrp)
    return pro.pid


def kill(pid, *args):
    print(f"Killing {pid}")
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    processes = [
        x
        for x in psutil.process_iter()
        if x.username() == os.environ["USER"] and len(x.cmdline()) > 1 and "/training.py" in x.cmdline()[1]
    ]
    for proc in processes:
        proc.kill()


def kill_and_exit(pid, *args):
    print(f"Killing {pid}")
    os.killpg(os.getpgid(pid), signal.SIGKILL)
    exit()


def terminate(pid, *args):
    print(f"Terminating {pid}")
    os.killpg(os.getpgid(pid), signal.SIGTERM)


def terminate_and_exit(pid, *args):
    print(f"Terminating {pid}")
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    exit()


def parse_errors(i):
    print(f"Check {error_file_name(i)} for error")
    error_lines = set()
    with open(error_file_name(i), "r") as f:
        try:
            for line in f.readlines():
                if "error" in line.lower() and "wandb: Network error" not in line:
                    error_lines.add(line)
        except UnicodeDecodeError:
            pass
    return error_lines


def pause():
    with open(f"{LOG_DIR / '.pause'}", "a"):
        pass


def resume():
    try:
        os.remove(f"{LOG_DIR / '.pause'}")
    except FileNotFoundError:
        print("file .pause does not exist, resuming anyway.")


def check_pause():
    return os.path.isfile(f"{LOG_DIR / '.pause'}")


def check_restart():
    if os.path.isfile(f"{LOG_DIR / '.restart'}"):
        os.remove(f"{LOG_DIR / '.restart'}")
        return True
    return False


def wait_for_error(i, previous_error, pid):
    print("Wait for error")
    current_errors = set()
    while not len(current_errors):
        current_errors = parse_errors(i)
        if check_restart():
            print("Restart commanded")
            break
        time.sleep(WAIT_INTERVAL)
    if len(current_errors):
        print("Found Error!")
    kill(pid)
    if len(current_errors) and previous_error == current_errors:
        pause()
        while check_pause():
            time.sleep(WAIT_INTERVAL)
    return current_errors


def start_training_loop(cmd):
    # do not write cmd if job is resubmitted (train_cmd.txt exists in that case)
    if get_train_cmd() is None:
        write_cmd_to_file(cmd)
    time.sleep(1)
    i = check_dir_for_previous_files()
    previous_error = set()
    while True:
        cmd = get_train_cmd()
        pid = run_training(i, cmd)
        signal.signal(signal.SIGTERM, partial(terminate_and_exit, pid))
        signal.signal(signal.SIGINT, partial(terminate_and_exit, pid))
        previous_error = wait_for_error(i, previous_error, pid)
        time.sleep(30)
        i += 1


def write_cmd_to_file(train_cmd):
    file_path = Path(__file__).absolute().parents[2] / "train_cmd.txt"
    with open(file_path, "w") as file:
        file.write(" ".join(train_cmd))


def get_train_cmd():
    file_path = Path(__file__).absolute().parents[2] / "train_cmd.txt"
    if not file_path.exists():
        return None
    with open(file_path, "r") as file:
        cmd = file.read().split()
        return cmd


if __name__ == "__main__":
    path = Path(sys.argv[0]).parent
    cmd = ["srun", "python", (path / "training.py").as_posix()] + sys.argv[1:] + ["~callbacks/shm_signal"]

    start_training_loop(cmd)
