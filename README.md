# HULC
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/mees/hulc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/mees/hulc/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/mees/hulc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/mees/hulc/alerts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[<b>What Matters in Language Conditioned Imitation Learning</b>](https://arxiv.org/pdf/2204.06252.pdf)

[Oier Mees](https://www.oiermees.com/), [Lukas Hermann](http://www2.informatik.uni-freiburg.de/~hermannl/), [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)

 We present **HULC** (**H**ierarchical **U**niversal **L**anguage **C**onditioned Policies), an end-to-end model that can
 learn  a wide variety of language conditioned robot skills from  offline free-form imitation datasets. HULC sets a new state of the art on the challenging CALVIN benchmark,
 on learning a single 7-DoF policy that can perform long-horizon manipulation tasks in a 3D environment, directly from images, and only specified with natural language.
This code accompanies the paper What Matters in Language Conditioned Imitation Learning, which can be found [here](https://arxiv.org/pdf/2204.06252.pdf).
We hope the code will be useful as a starting point for further research on language conditioned policy learning and will bring us closer towards general-purpose robots that can relate human language to their perception and actions.

![](media/hulc_rollout.gif)
## Installation
To begin, clone this repository locally
```bash
git clone --recurse-submodules https://github.com/mees/hulc.git
export HULC_ROOT=$(pwd)/hulc

```
Install requirements:
```bash
cd $HULC_ROOT
conda create -n hulc_venv python=3.8  # or use virtualenv
conda activate hulc_venv
sh install.sh
```
If you encounter problems installing pyhash, you might have to downgrade setuptools to a version below 58.

## Download
### CALVIN Dataset
If you want to train on the [CALVIN](https://github.com/mees/calvin) dataset, choose a split with:
```bash
cd $HULC_ROOT/dataset
sh download_data.sh D | ABC | ABCD | debug
```
If you want to get started without downloading the whole dataset, use the argument `debug` to download a small debug dataset (1.3 GB).
### Language Embeddings
We provide the precomputed embeddings of the different Language Models we evaluate in the paper.
The script assumes the corresponding split has been already downloaded.
```bash
cd $HULC_ROOT/dataset
sh download_lang_embeddings.sh D | ABC | ABCD
```

### Pre-trained Models
We provide our final models for all three CALVIN splits.
```bash
cd $HULC_ROOT/checkpoints
sh download_model_weights.sh D | ABC | ABCD
```
For instructions how to use the pretrained models, look at the training and evaluation sections.

## Hardware Requirements

We leverage [Pytorch Lightning's](https://www.pytorchlightning.ai/) DDP implementation to scale our training to 8x NVIDIA GPUs with **12GB** memory each.
Evaluating the models requires a single NVIDIA GPU with **8GB**. As each GPU receives a batch of 64 sequences (32 language + 32 vision), the effective batch size is 512 for all our experiments.

Trained with:
- **GPU** - 8x NVIDIA RTX 2080Ti
- **CPU** - AMD EPYC 7502
- **RAM** - 512GB
- **OS** - Ubuntu 20.04

## Training
To train our HULC model with the maximum amount of available GPUS, run:
```
python hulc/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset datamodule/datasets=vision_lang_shm
```
The `vision_lang_shm` option loads the CALVIN dataset into shared memory at the beginning of the training,
speeding up the data loading during training.
The preparation of the shared memory cache will take some time
(approx. 20 min at our SLURM cluster). \
If you want to use the original data loader (e.g. for debugging) just override the command with `datamodule/datasets=vision_lang`. \
For an additional speed up, you can disable the evaluation callbacks during training by adding `~callbacks/rollout` and `~callbacks/rollout_lh`

If you have access to a SLURM cluster, follow this [guide](https://github.com/mees/hulc/blob/main/slurm_scripts/README.md).

You can use our [pre-trained models](#pre-trained-models) to initialize a training by running
```
python hulc/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset hydra.run.dir=$HULC_ROOT/checkpoints/HULC_D_D
```
Note that this will log the training into the checkpoint folder.

### Ablations
Multi-context imitation learning (MCIL), (Lynch et al., 2019):
```
python hulc/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset datamodule/datasets=vision_lang_shm model=mcil
datamodule=mcil
```

Goal-conditioned behavior cloning (GCBC), (Lynch et al., 2019):
```
python hulc/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset datamodule/datasets=vision_lang_shm model=gcbc
~callbacks/tsne_plot
```


## Evaluation
See detailed inference instructions on the [CALVIN repo](https://github.com/mees/calvin#muscle-evaluation-the-calvin-challenge).
```
python hulc/evaluation/evaluate_policy.py --dataset_path <PATH/TO/DATASET> --train_folder <PATH/TO/TRAINING/FOLDER>
```
Set `--train_folder $HULC_ROOT/checkpoints/HULC_D_D` to evaluate our [pre-trained models](#pre-trained-models).

Optional arguments:

- `--checkpoint <PATH/TO/CHECKPOINT>`: by default, the evaluation loads the last checkpoint in the training log directory.
You can instead specify the path to another checkpoint by adding this to the evaluation command.
- `--debug`: print debug information and visualize environment.

## Acknowledgements

This work uses code from the following open-source projects and datasets:

#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### Sentence-Transformers
Original:  [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
License: [Apache 2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)

#### OpenAI CLIP
Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)
## Citations

If you find the code useful, please cite:

**HULC**
```bibtex
@article{hulc22,
author = {Oier Mees and Lukas Hermann and Wolfram Burgard},
title = {What Matters in Language Conditioned Imitation Learning},
journal={arXiv preprint arXiv:2204.06252},
year = 2022,
}
```
**CALVIN**
```bibtex
@article{calvin21,
author = {Oier Mees and Lukas Hermann and Erick Rosete-Beas and Wolfram Burgard},
title = {CALVIN: A benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks},
journal={arXiv preprint arXiv:2112.03227},
year = 2021,
}
```

## License

MIT License
