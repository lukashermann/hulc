import logging
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional

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


class ShmDataset(BaseDataset):
    """
    Dataset Loader that uses a shared memory cache

    parameters
    ----------

    datasets_dir:       path of folder containing episode files (string must contain 'validation' or 'training')
    save_format:        format of episodes in datasets_dir (.pkl or .npz)
    obs_space:          DictConfig of the observation modalities of the dataset
    max_window_size:    maximum length of the episodes sampled from the dataset
    """

    def __init__(self, *args, skip_frames: int = 0, aux_lang_loss_window: int = 1, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
        self.skip_frames = skip_frames
        self.aux_lang_loss_window = aux_lang_loss_window
        self.episode_lookup_dict: Dict[str, List] = {}
        self.episode_counters: Optional[np.ndarray] = None
        self.lang_lookup = None
        self.lang_ann = None
        self.shapes = None
        self.sizes = None
        self.dtypes = None
        self.dataset_type = None
        self.shared_memories = None

    def set_lang_data(self, lang_data):
        if self.with_lang:
            self.episode_lookup_dict = lang_data["episode_lookup_lang"]
            self.lang_lookup = lang_data["lang_lookup"]
            self.lang_ann = lang_data["lang_ann"]
        else:
            self.episode_lookup_dict = lang_data["episode_lookup_vision"]
        key = list(self.episode_lookup_dict.keys())[0]
        self.episode_counters = np.array(self.episode_lookup_dict[key])[:, 1]
        self.shapes = lang_data["shapes"]
        self.sizes = lang_data["sizes"]
        self.dtypes = lang_data["dtypes"]
        self.dataset_type = "train" if "training" in self.abs_datasets_dir.as_posix() else "val"
        # attach to shared memories
        self.shared_memories = {
            key: SharedMemory(name=f"{self.dataset_type}_{key}") for key in self.episode_lookup_dict
        }

    # @rank_zero_only
    # def __del__(self):
    #     if self.with_lang:
    #         for shm in self.shared_memories.values():
    #             try:
    #                 shm.close()
    #                 shm.unlink()
    #             except FileNotFoundError:
    #                 pass

    def get_window_size(self, idx):
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_counters) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_counters) - idx - 1
        elif self.episode_counters[idx + window_diff] != self.episode_counters[idx] + window_diff:
            # less than max_episode steps until next episode
            steps_to_next_episode = (
                self.min_window_size
                + np.nonzero(
                    self.episode_counters[idx : idx + window_diff + 1]
                    - (self.episode_counters[idx] + np.arange(window_diff + 1))
                )[0][0]
                - 1
            )
            max_window = min(self.max_window_size, int(steps_to_next_episode))
        else:
            max_window = self.max_window_size

        if self.validation:
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def load_sequence_shm(self, idx, window_size):
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
        episode = {}
        for key, lookup in self.episode_lookup_dict.items():
            offset, j = lookup[idx]
            shape = (window_size + j,) + self.shapes[key]
            array = np.ndarray(shape, dtype=self.dtypes[key], buffer=self.shared_memories[key].buf, offset=offset)[j:]
            episode[key] = array
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
        return episode

    def get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        parameters
        ----------
        idx: index of starting frame
        window_size:    length of sampled episode

        returns
        ----------
        seq_state_obs:  numpy array of state observations
        seq_rgb_obs:    tuple of numpy arrays of rgb observations
        seq_depth_obs:  tuple of numpy arrays of depths observations
        seq_acts:       numpy array of actions
        """
        episode = self.load_sequence_shm(idx, window_size)

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        info = self.add_language_info(info, idx)
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_depth_obs, **seq_acts, **info, **seq_lang}  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def add_language_info(self, info, idx):
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window < len(self.lang_lookup)
            and self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

    def __len__(self):
        """
        returns
        ----------
        number of possible starting frames
        """
        return len(list(self.episode_lookup_dict.values())[0])
