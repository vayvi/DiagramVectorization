from typing import Any, Optional
import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.optim.lr_scheduler as lr_scheduler
import math
from helper.misc import (
    reduce_dict,
)
import os
import subprocess
from pathlib import Path


class BaseModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def on_test_start(self) -> None:
        output_dir = self.hparams.output_dir
        if self.hparams.letr_cfg.eval_real:  # TODO: change this to actual folder name
            benchmark_folder = f"real_val/epoch_{self.hparams.previous_epoch}"
        else:
            benchmark_folder = f"synthetic_val/epoch_{self.hparams.previous_epoch}"

        npz_dir = Path(output_dir) / benchmark_folder
        os.makedirs(npz_dir)
        self.hparams.npz_dir = str(npz_dir)

    def allsplit_step(self, split: str, batch, batch_idx):
        sample, target = batch
        out = self.forward(sample)  # forward pass

        loss_dict = self.criterion(
            out,
            target,
        )  # compute loss lines and circles
        weight_dict = self.criterion.weight_dict

        losses_log = process_loss_dict(loss_dict, weight_dict)
        self.log_dict(losses_log, batch_size=self.hparams.letr_cfg.batch_size)

        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        del loss_dict, weight_dict
        self.log(
            f"{split}_step_loss", loss, batch_size=self.hparams.letr_cfg.batch_size
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):  # TODO: Add post step support
        sample, target = batch
        outputs = self.forward(sample)  # forward pass
        orig_target_sizes = torch.stack([t["orig_size"] for t in target], dim=0)
        postprocessor = self.postprocessors["line"]  # FIXME: only line for now
        primitive = "shapes"
        outputs = outputs[primitive]
        results = postprocessor(outputs, orig_target_sizes)
        pred_logits = outputs["pred_logits"]

        bz, query = pred_logits.shape[0], pred_logits.shape[1]
        assert bz == 1, "only support batch size 1"

        rst = results[0][primitive]

        pred_lines = rst.view(query, 2, 2)
        pred_lines = pred_lines.flip([-1])  # this is yxyx format
        h, w = target[0]["orig_size"].tolist()

        pred_lines[:, :, 0] = pred_lines[:, :, 0] * (128 / h)
        pred_lines[:, :, 1] = pred_lines[:, :, 1] * (128 / w)

        score = results[0]["scores"].cpu().numpy()
        label = results[0]["labels"].cpu().numpy()

        pred_lines = pred_lines.flip([-1])  # this is yxyx format
        line = pred_lines.cpu().numpy()
        score_idx = np.argsort(-score)
        line, score, label = line[score_idx], score[score_idx], label[score_idx]

        img_id = int(target[0]["image_id"].cpu().numpy())

        checkpoint_path = os.path.join(
            self.hparams.npz_dir, f"{str(self.hparams.id_to_img[img_id])}.npz"
        )
        np.savez(
            checkpoint_path,
            **{primitive: line, "score": score, "label": label},
        )

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.hparams.backbone_cfg.lr_backbone,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            param_dicts,  # TODO: check how to freeze l certain params
            lr=self.hparams.letr_cfg.optimizer.lr,
            weight_decay=self.hparams.letr_cfg.optimizer.weight_decay,
        )
        self.lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=self.hparams.letr_cfg.optimizer.lr_drop
        )
        return [self.optimizer], {
            "scheduler": self.lr_scheduler,
        }

    @torch.jit.unused
    def _set_aux_loss(self, class_pred, primitive_pred, loss_name="pred_shapes"):
        return [
            {"pred_logits": a, loss_name: b} for a, b in zip(class_pred, primitive_pred)
        ]


def process_loss_dict(loss_dict, weight_dict):
    loss_dict_reduced = reduce_dict(loss_dict)

    loss_dict_reduced_scaled = {
        k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
    }
    losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

    loss_value = losses_reduced_scaled.item()

    if not math.isfinite(loss_value):
        raise ValueError(
            f"### Loss:{loss_value}, stopping training. loss dict:{loss_dict_reduced}"
        )
    return {
        "loss": loss_value,
        **loss_dict_reduced,
    }
