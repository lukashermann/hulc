from collections import Counter, defaultdict
from functools import partial
from itertools import chain
import logging
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import re
import signal

import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from tqdm import tqdm

from hulc.datasets.shm_dataset import ShmDataset

log = logging.getLogger(__name__)


def gather_results(return_dict):
    episode_lookup_vision = defaultdict(list)
    lang_episode_dict = defaultdict(dict)
    for proc in sorted(return_dict):
        for key in return_dict[proc][0]:
            episode_lookup_vision[key] += return_dict[proc][0][key]
            lang_episode_dict[key].update(return_dict[proc][1][key])
    return episode_lookup_vision, lang_episode_dict


def check_shm_lookup_exists(dataset_type):
    load_path = Path("/tmp/") if "TMPDIR" not in os.environ else Path(os.environ["TMPDIR"])
    try:
        data = np.load(load_path / f"{dataset_type}_shm_lookup.npy", allow_pickle=True).item()
        return data
    except FileNotFoundError:
        return None


class SharedMemoryLoader:
    def __init__(self, datasets_cfg, dataset_dir):

        self.obs_space = datasets_cfg.lang_dataset.obs_space
        self.dataset_dir = dataset_dir
        self.dataset_type = "train" if "training" in dataset_dir.as_posix() else "val"
        self.lang_folder = datasets_cfg.lang_dataset.lang_folder
        self.save_format = "npz"
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()
        self.min_window_size_vision = datasets_cfg.vision_dataset.min_window_size
        self.min_window_size_lang = datasets_cfg.lang_dataset.min_window_size
        self.n_proc = 8

    def worker_process(self, proc_num, ep_start_end_ids, offsets, shmem, lang_ep_start_end_ids, return_dict):
        episode_lookup_vision = defaultdict(list)
        lang_episode_dict = defaultdict(dict)
        if proc_num == 0:
            pbar = tqdm(total=np.sum(np.diff(ep_start_end_ids)), leave=False)
        else:
            pbar = None
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            seq = self.zip_sequence(start_idx, end_idx, pbar)
            for key, array in seq.items():
                shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shmem[key].buf, offset=offsets[key])
                shared_array[:] = array[:]

                for j, idx in enumerate(range(start_idx, end_idx + 1 - self.min_window_size_vision)):
                    episode_lookup_vision[key].append((offsets[key], j))
                    if idx in lang_ep_start_end_ids[:, 0]:
                        lang_episode_dict[key][idx] = (offsets[key], j)
                offsets[key] += array.nbytes
        return_dict[proc_num] = episode_lookup_vision, lang_episode_dict
        if pbar is not None:
            pbar.close()

    def load_data_in_shared_memory(self):
        lang_data = np.load(self.dataset_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        ep_start_end_ids = np.load(self.dataset_dir / "ep_start_end_ids.npy")
        lang_ep_start_end_ids = np.array(lang_data["info"]["indx"])  # each of them are 64
        lang_ann = lang_data["language"]["emb"]
        shmem, shapes, sizes, dtypes, shmem_lookup = self.create_shmem(ep_start_end_ids)

        if shmem_lookup is not None:
            # using existing shared memory
            log.info("Using existing shared memory without reloading it.")
            return shmem_lookup

        lang_lookup = []

        episode_lookup_lang = defaultdict(list)
        log.info(
            f"Loading {self.dataset_type} language episodes into shared memory. "
            f"(progress bar shows only worker process 0)."
        )

        if self.n_proc > len(ep_start_end_ids):
            self.n_proc = len(ep_start_end_ids)
        split_indices = np.array_split(ep_start_end_ids, self.n_proc, axis=0)
        split_lens = [np.sum(np.diff(split_indices[i])) for i in range(len(split_indices))]
        obs_size = {key: dtypes[key].itemsize * np.prod(shapes[key]) for key in dtypes}
        offsets = [{key: n * obs_size[key] for key in dtypes} for n in np.cumsum([0] + split_lens[:-1])]

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []
        # load vision data with multiple processes
        for i in range(self.n_proc):
            p = multiprocessing.Process(
                target=self.worker_process,
                args=(i, split_indices[i], offsets[i], shmem, lang_ep_start_end_ids, return_dict),
            )
            processes.append(p)
            p.start()
        for proc in processes:
            proc.join()

        episode_lookup_vision, lang_episode_dict = gather_results(return_dict)

        # lang data
        for i, (start_idx, end_idx) in enumerate(tqdm(lang_ep_start_end_ids)):
            for key in lang_episode_dict:
                offset, step = lang_episode_dict[key][start_idx]
                for j, idx in enumerate(range(start_idx, end_idx + 1 - self.min_window_size_lang)):
                    episode_lookup_lang[key].append((offset, step + j))
            for idx in range(start_idx, end_idx + 1 - self.min_window_size_lang):
                lang_lookup.append(i)
        result = {
            "episode_lookup_vision": episode_lookup_vision,
            "episode_lookup_lang": episode_lookup_lang,
            "lang_lookup": lang_lookup,
            "lang_ann": lang_ann,
            "shapes": shapes,
            "sizes": sizes,
            "dtypes": dtypes,
        }
        return result

    def create_shmem(self, ep_start_end_ids):
        # load first episode to determine memory usage
        seq = self.zip_sequence(ep_start_end_ids[0][0], ep_start_end_ids[0][0] + 1)
        total_size = np.sum(ep_start_end_ids[:, 1] - ep_start_end_ids[:, 0])
        shmem = {}
        shapes = {}
        sizes = {}
        dtypes = {}

        shm_lookup = check_shm_lookup_exists(self.dataset_type)
        # check if all necessary shared memories are already loaded
        if shm_lookup is not None:
            print("shm_lookup exists")
            try:
                if np.all(
                    [
                        SharedMemory(name=f"{self.dataset_type}_{key}").size == size * total_size
                        for key, size in shm_lookup["sizes"].items()
                    ]
                ):
                    return None, None, None, None, shm_lookup
            except FileNotFoundError as e:
                pass
        for key, array in seq.items():
            try:
                # see if exists
                s = SharedMemory(name=f"{self.dataset_type}_{key}")
                s.close()
                s.unlink()
                log.warning(
                    f"Found existing shared memory {self.dataset_type}_{key}, freeing up memory."
                    "In case of multiple training runs on the same node, this will lead to problems."
                )
            except FileNotFoundError:
                pass
            shmem[key] = SharedMemory(create=True, size=array.nbytes * total_size, name=f"{self.dataset_type}_{key}")
            shapes[key] = array.shape[1:]
            sizes[key] = array.nbytes
            dtypes[key] = array.dtype

        # register signal handler for the case that shm data loading process gets interrupted.
        signal.signal(signal.SIGTERM, partial(delete_shm, shmem.keys()))

        return shmem, shapes, sizes, dtypes, None

    def zip_sequence(self, start_idx, end_idx, pbar=None):
        keys = list(chain(*self.obs_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        n_items = end_idx - start_idx
        episode = {}
        data = np.load(self.get_episode_name(start_idx))
        for key in keys:
            shape = (n_items,) + data[key].shape
            dtype = data[key].dtype
            episode[key] = np.empty(shape=shape, dtype=dtype)
        for i, file_idx in enumerate(range(start_idx, end_idx)):
            with np.load(self.get_episode_name(file_idx)) as data:
                for key in keys:
                    episode[key][i] = data[key]
            if pbar is not None:
                pbar.update(1)
        return episode

    def get_episode_name(self, idx):
        """
        Convert frame idx to file name
        """
        return Path(f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def lookup_naming_pattern(self, n_digits=None):
        it = os.scandir(self.dataset_dir)
        while True:
            filename = Path(next(it))
            if self.save_format in filename.suffix:
                break
        aux_naming_pattern = re.split(r"\d+", filename.stem)
        naming_pattern = [filename.parent / aux_naming_pattern[0], filename.suffix]
        n_digits = n_digits if n_digits is not None else len(re.findall(r"\d+", filename.stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits


def delete_shm(shm_keys, signal, frame):
    for dataset_type in ["train", "val"]:
        for shm_key in shm_keys:
            try:
                s = SharedMemory(name=f"{dataset_type}_{shm_key}")
                s.close()
                s.unlink()
                print(f"successfully unlinked {shm_key}")
            except Exception as e:
                print(e)
    exit()


class SignalCallback(Callback):
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if isinstance(trainer.datamodule.train_dataloader()["vis"].dataset, ShmDataset):  # type: ignore
            shm_keys = trainer.datamodule.train_dataloader()["vis"].dataset.episode_lookup_dict.keys()  # type: ignore
            signal.signal(signal.SIGTERM, partial(delete_shm, shm_keys))
            print("Registered shared memory signal handler.")
