import torch
import torch.nn.functional as F
from torch import nn
from helper.misc import (
    nested_tensor_from_tensor_list,
)
from LETR.model.backbone import build_backbone
from LETR.model.transformer.transformer import build_transformer
from LETR.model.losses.losses import SetCriterion
from LETR.model.utils.postprocessors import PostProcess_Line
from .base_letr import BaseModel


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LETR(BaseModel):
    def __init__(
        self,
        backbone_cfg,
        position_cfg,
        transformer_stage1_cfg,
        criterion_cfg,
        letr_cfg,
        transformer_stage2_cfg=None,
        id_to_img=None,
    ):
        super().__init__()
        self.save_hyperparameters("letr_cfg", ignore="id_to_img")

        print("Initializing backbone")
        self.backbone = build_backbone(backbone_cfg, position_cfg)
        print("Initializing base transformer")
        self.transformer = build_transformer(transformer_stage1_cfg)
        if letr_cfg.layer1_frozen:
            print("Freezing backbone and transformer_stage1")
            for n, p in self.named_parameters():
                p.requires_grad_(False)

        print("Initializing losses")
        self.criterion = SetCriterion(criterion_cfg)
        self.names = letr_cfg.names  # for line only or circle only support as well
        self.letr_cfg = letr_cfg
        print("Initializing postprocessors")
        self.postprocessors = {"line": PostProcess_Line(self.letr_cfg.output_type)}
        self.channel = [256, 512, 1024, 2048]
        hidden_dim = self.transformer.d_model
        self.num_queries = self.letr_cfg.num_queries
        self.class_embed_lines = nn.Linear(hidden_dim, self.letr_cfg.num_classes + 1)
        self.query_embed_lines = nn.Embedding(self.num_queries, hidden_dim)
        self.lines_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.queries = [self.query_embed_lines.weight]
        self.class_embeds = [self.class_embed_lines]
        self.primitive_embeds = [self.lines_embed]

        self.layer1_num = self.letr_cfg.layer1_num
        self.input_proj = nn.Conv2d(
            self.channel[self.layer1_num], hidden_dim, kernel_size=1
        )
        if self.letr_cfg.LETRpost:
            self.layer2_num = self.letr_cfg.layer2_num
            self.transformer_stage2 = build_transformer(transformer_stage2_cfg)
            self.input_proj2 = nn.Conv2d(
                self.channel[self.layer2_num], hidden_dim, kernel_size=1
            )

        self.__post_init__()

    def forward(self, samples):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos_embed = self.backbone(samples)
        src, mask = features[self.letr_cfg.layer1_num].decompose()
        assert mask is not None, "mask is None"

        out = {}

        hs = self.transformer(
            self.input_proj(src),
            mask,
            self.queries,
            pos_embed[self.layer1_num],
        )

        # stage 2
        if self.letr_cfg.LETRpost:
            src2, mask2 = features[self.layer2_num].decompose()
            src2 = self.input_proj2(src2)
            query = [hs[-1]]
            hs = self.transformer_stage2(src2, mask2, query, pos_embed[self.layer2_num])

        hs_primitives = [
            hs[:, :, : self.num_queries, :],
            hs[:, :, self.num_queries :, :],
        ]
        for hs_primitive, class_embed, primitive_embed, name in zip(
            hs_primitives, self.class_embeds, self.primitive_embeds, self.names
        ):
            loss_name = f"pred_{name}"

            class_pred = class_embed(hs_primitive)
            pred = primitive_embed(hs_primitive).sigmoid()
            out[name] = {"pred_logits": class_pred[-1], loss_name: pred[-1]}
            # out = {"pred_logits": class_pred[-1], loss_name: pred[-1]}

            if self.letr_cfg.criterion.aux_loss:
                # out["aux_outputs"] = self._set_aux_loss(class_pred, pred, loss_name=loss_name)
                out[name]["aux_outputs"] = self._set_aux_loss(
                    class_pred, pred, loss_name=loss_name
                )

        return out
