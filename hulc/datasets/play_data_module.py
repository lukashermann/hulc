import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
import torchvision

import hulc
from hulc.datasets.utils.episode_utils import load_dataset_statistics
from hulc.datasets.utils.shared_memory_loader import SharedMemoryLoader

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})
ONE_EP_DATASET_URL = "http://www.informatik.uni-freiburg.de/~meeso/50steps.tar.xz"


class PlayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        root_data_dir: str = "data",
        num_workers: int = 8,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            root_data_path = Path(hulc.__file__).parent / root_data_path
        self.training_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"
        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []
        self.transforms = transforms

        self.use_shm = "shm_dataset" in self.datasets_cfg.lang_dataset._target_

    def prepare_data(self, *args, **kwargs):
        # check if files already exist
        dataset_exist = np.any([len(list(self.training_dir.glob(extension))) for extension in ["*.npz", "*.pkl"]])

        # download and unpack images
        if not dataset_exist:
            logger.info(f"downloading dataset to {self.training_dir} and {self.val_dir}")
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.training_dir)
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.val_dir)

        if self.use_shm:
            train_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.training_dir)
            train_shm_lookup = train_shmem_loader.load_data_in_shared_memory()

            val_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.val_dir)
            val_shm_lookup = val_shmem_loader.load_data_in_shared_memory()

            save_lang_data(train_shm_lookup, val_shm_lookup)

    def setup(self, stage=None):
        transforms = load_dataset_statistics(self.training_dir, self.val_dir, self.transforms)

        self.train_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.train[cam]] for cam in transforms.train
        }

        self.val_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.val[cam]] for cam in transforms.val
        }
        self.train_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.train_transforms.items()}
        self.val_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.val_transforms.items()}
        self.train_datasets, self.train_sampler, self.val_datasets, self.val_sampler = {}, {}, {}, {}

        if self.use_shm:
            train_shm_lookup, val_shm_lookup = load_lang_data()

        for _, dataset in self.datasets_cfg.items():
            train_dataset = hydra.utils.instantiate(
                dataset, datasets_dir=self.training_dir, transforms=self.train_transforms
            )
            val_dataset = hydra.utils.instantiate(dataset, datasets_dir=self.val_dir, transforms=self.val_transforms)
            if self.use_shm:
                train_dataset.set_lang_data(train_shm_lookup)
                val_dataset.set_lang_data(val_shm_lookup)
            key = dataset.key
            self.train_datasets[key] = train_dataset
            self.val_datasets[key] = val_dataset
            self.modalities.append(key)

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=False,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        val_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=False,
            )
            for key, dataset in self.val_datasets.items()
        }
        combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        return combined_val_loaders


def save_lang_data(train_shm_lookup, val_shm_lookup):
    save_path = Path("/tmp/") if "TMPDIR" not in os.environ else Path(os.environ["TMPDIR"])
    np.save(save_path / "train_shm_lookup.npy", train_shm_lookup)
    np.save(save_path / "val_shm_lookup.npy", val_shm_lookup)


def load_lang_data():
    load_path = Path("/tmp/") if "TMPDIR" not in os.environ else Path(os.environ["TMPDIR"])
    train_shm_lookup = np.load(load_path / "train_shm_lookup.npy", allow_pickle=True).item()
    val_shm_lookup = np.load(load_path / "val_shm_lookup.npy", allow_pickle=True).item()
    return train_shm_lookup, val_shm_lookup
