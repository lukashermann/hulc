from pathlib import Path

from calvin_agent.utils.utils import format_sftp_path
from pytorch_lightning.utilities.cloud_io import load as pl_load


def initialize_pretrained_weights(model, cfg):
    pretrain_chk = pl_load(format_sftp_path(Path(cfg.pretrain_chk)), map_location=lambda storage, loc: storage)
    batch_size = model.plan_recognition.position_embeddings.weight.shape[0]
    weight = "plan_recognition.position_embeddings.weight"
    pretrain_chk["state_dict"][weight] = pretrain_chk["state_dict"][weight][:batch_size]
    if "pretrain_exclude_pr" in cfg and cfg.pretrain_exclude_pr:
        for key in list(pretrain_chk["state_dict"].keys()):
            if key.startswith("plan_recognition"):
                del pretrain_chk["state_dict"][key]
    model.load_state_dict(pretrain_chk["state_dict"], strict=False)
