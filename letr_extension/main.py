import torch
import shutil
import argparse
from glob import glob
import yaml
from types import SimpleNamespace
import re
from pathlib import Path
import wandb
from pytorch_lightning.loggers import WandbLogger
from LETR.model.letr import LETR

from LETR.data import build_dataset
import pytorch_lightning as pl
from helper.misc import collate_fn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import json


def get_id_to_img(args):
    id_to_img = {}
    mode = args.mode
    path_val = os.path.join(args.coco_path, "annotations", f"{mode}_val.json")
    with open(path_val) as f_val:
        data = json.load(f_val)
        for d in data["images"]:
            id_to_img[d["id"]] = d["file_name"].split(".")[0]
    with open(os.path.join(args.coco_path, "id_to_img.json"), "w") as f:
        json.dump(id_to_img, f)

    return id_to_img


def get_state_dict_from_url(url, named_parameters, named_buffers, layer1_num):
    print("resume from url")
    checkpoint = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", check_hash=True
    )
    new_state_dict = {}
    for k in checkpoint["model"]:
        if ("class_embed" in k) or ("bbox_embed" in k) or ("query_embed" in k):
            continue
        if ("input_proj" in k) and layer1_num != 3:
            continue
        new_state_dict[k] = checkpoint["model"][k]

    current_param = [n for n, p in named_parameters]
    current_buffer = [n for n, p in named_buffers]
    load_param = new_state_dict.keys()
    for p in load_param:
        if p not in current_param and p not in current_buffer:
            print(p, "NOT appear in current model. ")
    for p in current_param:
        if p not in load_param:
            print(p, "NEW parameter.")
    return new_state_dict


def get_state_dict_from_stage1(ckpt_path, named_parameters, named_buffers, layer1_num):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    transformer_state_dict = {}
    for k in checkpoint["state_dict"]:  # FIXME: support "model" and "state_dict"
        if "transformer" in k:
            new_key = k.replace("transformer.", "transformer_stage2.")
            transformer_state_dict[new_key] = checkpoint["state_dict"][k]
    new_state_dict = checkpoint["state_dict"]
    new_state_dict.update(transformer_state_dict)
    current_param = [n for n, p in named_parameters]
    current_buffer = [n for n, p in named_buffers]
    load_param = new_state_dict.keys()
    for p in load_param:
        if p not in current_param and p not in current_buffer:
            print(p, "NOT appear in current model. ")

    for p in current_param:
        if p not in load_param:
            print(p, "NEW parameter.")

    return new_state_dict


def main(config_path, epochs, test, coco_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = dict_to_namespace(config_dict)
    cfg.trainer.epochs = epochs
    cfg.test = test
    cfg.data.coco_path = coco_path
    if ("diagrams" in cfg.data.coco_path) or ("real" in cfg.data.coco_path):
        print("EVALUATING ON REAL DATA")
        cfg.letr.eval_real = True
        cfg.checkpoint.resume = "local"
    else:
        cfg.letr.eval_real = False
    print("Initializing data")
    if len(cfg.letr.names) > 1:
        cfg.letr.num_classes = 1
        cfg.letr.criterion.num_classes = 1
    print(cfg.letr.names, cfg.letr.criterion.names)
    assert cfg.letr.batch_size == cfg.data.batch_size

    data_loader_val = DataLoader(
        build_dataset("val", cfg.data),
        batch_size=1,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=cfg.trainer.num_workers,
        shuffle=False,
    )

    print("Initializing model")
    model = LETR(
        backbone_cfg=cfg.letr.backbone,
        position_cfg=cfg.letr.position_encoding,
        transformer_stage1_cfg=cfg.letr.transformer_stage1,
        criterion_cfg=cfg.letr.criterion,
        letr_cfg=cfg.letr,
        transformer_stage2_cfg=cfg.letr.transformer_stage2,
        id_to_img=get_id_to_img(cfg.data),
    )

    Path(cfg.checkpoint.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(cfg.checkpoint.output_dir) / "checkpoints"
    ckpt_path = None
    if cfg.checkpoint.resume:
        if cfg.checkpoint.resume.startswith("https"):
            new_state_dict = get_state_dict_from_url(
                cfg.checkpoint.resume,
                model.named_parameters(),
                model.named_buffers(),
                cfg.letr.layer1_num,
            )
            model.load_state_dict(new_state_dict, strict=False)
        elif cfg.checkpoint.resume == ("local"):
            print("Resume from local checkpoint, model and optimizer ")
            print(output_dir)
            for ckpt_path in sorted(glob(str(output_dir / "*.ckpt"))):
                print("found checkpoint: ", ckpt_path)
            ckpt_callback = None
        else:
            print(f"Resume from local checkpoint: {cfg.checkpoint.resume}")
            if cfg.checkpoint.load_ckpt:
                ckpt_path = cfg.checkpoint.resume
                print("successfully loaded ckpt")
            else:
                new_state_dict = get_state_dict_from_stage1(
                    cfg.checkpoint.resume,
                    model.named_parameters(),
                    model.named_buffers(),
                    cfg.letr.layer1_num,
                )
                model.load_state_dict(new_state_dict, strict=False)
                print("successfully loaded state dict")

    else:
        print("No checkpoint found. Training from scratch")
        output_dir.mkdir(parents=True)
    ckpt = ModelCheckpoint(dirpath=output_dir, filename="model-{epoch:02d}", mode="min")
    ckpt_callback = [ckpt]

    print("Initializing Trainer")

    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, config=config_dict)
        logger = WandbLogger()
    else:
        logger = None
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=logger,
        max_epochs=cfg.trainer.epochs,
        callbacks=ckpt_callback,
        gradient_clip_val=cfg.trainer.clip_max_norm,
        precision=cfg.trainer.precision,
    )

    print("Starting training")
    if cfg.test:
        print("Starting testing")
        print("Using checkpoint: ", ckpt_path)

        epoch_pattern = r"model-epoch=(\d+)\.ckpt"
        match = re.search(epoch_pattern, ckpt_path)
        # Get the epoch number as an integer
        if match:
            model.hparams.previous_epoch = str(match.group(1))
            print(str(match.group(1)))
        else:
            print("No epoch number found in the checkpoint path.")
            model.hparams.previous_epoch = "unknown"
        model.hparams.output_dir = str(cfg.checkpoint.output_dir)
        print(model.hparams.output_dir)
        os.makedirs(cfg.checkpoint.output_dir, exist_ok=True)

        trainer.test(model, dataloaders=data_loader_val, ckpt_path=ckpt_path)
    else:
        data_loader_train = DataLoader(
            build_dataset("train", cfg.data),
            batch_size=cfg.data.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.trainer.num_workers,
            drop_last=True,
            shuffle=True,
        )
        print("Starting training")
        trainer.fit(
            model,
            train_dataloaders=data_loader_train,
            val_dataloaders=data_loader_val,
            ckpt_path=ckpt_path,
        )
        shutil.copyfile(config_path, output_dir / "config_primitives.yaml")


def dict_to_namespace(dictionary):
    d = dictionary
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    nargs="?",
    default="config.yaml",
    help="Path to the config file",
)
parser.add_argument(
    "--coco_path",
    type=str,
    nargs="?",
    default="data/synthetic_processed",
    help="Path to the config file",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="If true, run test instead of training",
)
parser.add_argument(
    "--epochs",
    type=int,
    nargs="?",
    default=400,
    help="Number of epochs to train for",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.config_path, args.epochs, args.test, args.coco_path)
