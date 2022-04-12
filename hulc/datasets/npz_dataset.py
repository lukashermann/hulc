from itertools import chain
import logging
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from hulc.datasets.base_dataset import BaseDataset, get_validation_window_size
from hulc.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
)

logger = logging.getLogger(__name__)


class NpzDataset(BaseDataset):
    """
    Dataset Loader that uses a shared memory cache

    parameters
    ----------

    datasets_dir:       path of folder containing episode files (string must contain 'validation' or 'training')
    save_format:        format of episodes in datasets_dir (.pkl or .npz)
    obs_space:          DictConfig of the observation modalities of the dataset
    max_window_size:    maximum length of the episodes sampled from the dataset
    """

    def __init__(self, *args, skip_frames: int = 0, n_digits: Optional[int] = None, aux_lang_loss_window: int = 1, pretrain: bool = False, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
        self.skip_frames = skip_frames
        self.aux_lang_loss_window = aux_lang_loss_window
        self.pretrain = pretrain
        if self.with_lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann = self.load_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self.load_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = self.lookup_naming_pattern(n_digits)

    def lookup_naming_pattern(self, n_digits):
        it = os.scandir(self.abs_datasets_dir)
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

    def get_window_size(self, idx: int) -> int:
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif self.episode_lookup[idx + window_diff] != self.episode_lookup[idx] + window_diff:
            # less than max_episode steps until next episode
            steps_to_next_episode = (
                self.min_window_size
                + np.nonzero(
                    np.array(self.episode_lookup[idx : idx + window_diff + 1])
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
                - 1
            )
            max_window = min(self.max_window_size, steps_to_next_episode)
        else:
            max_window = self.max_window_size

        if self.validation:
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def get_episode_name(self, idx: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def zip_sequence(self, start_idx: int, end_idx: int, idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive individual frames saved as npy files and combine to episode dict
        parameters:
        -----------
        start_idx: index of first frame
        end_idx: index of last frame

        returns:
        -----------
        episode: dict of numpy arrays containing the episode where keys are the names of modalities
        """
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [self.load_episode(self.get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
        return episode

    def get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        Load sequence of length window_size.
        Args:
            idx: index of starting frame
            window_size: length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """
        start_file_indx = self.episode_lookup[idx]
        end_file_indx = start_file_indx + window_size

        episode = self.zip_sequence(start_file_indx, end_file_indx, idx)

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        # seq_lang = {"lang": torch.from_numpy(episode["language"]) if self.with_lang else torch.empty(0)}
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        info = self.add_language_info(info, idx)
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_depth_obs, **seq_acts, **info, **seq_lang}  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def load_file_indices_lang(self, abs_datasets_dir: Path) -> Tuple[List, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset

        Returns:
            episode_lookup: List for the mapping from training example index to episode (file) index
            lang_lookup:
            lang_ann:
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print("trying to load lang data from: ", abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).reshape(
                -1
            )[0]
        except Exception:
            print("Exception, trying to load lang data from: ", abs_datasets_dir / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).reshape(-1)[0]

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["emb"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.pretrain:
                start_idx = max(start_idx, end_idx + 1 - self.min_window_size - self.aux_lang_loss_window)
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return episode_lookup, lang_lookup, lang_ann

    def load_file_indices(self, abs_datasets_dir: Path) -> List:
        """
        this method builds the mapping from index to file_name used for loading the episodes

        parameters
        ----------
        abs_datasets_dir:               absolute path of the directory containing the dataset

        returns
        ----------
        episode_lookup:                 list for the mapping from training example index to episode (file) index
        max_batched_length_per_demo:    list of possible starting indices per episode
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return episode_lookup

    def add_language_info(self, info, idx):
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info
