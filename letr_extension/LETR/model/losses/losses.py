from torch import nn
import torch
import torch
import torch.nn.functional as F
from torch import nn
from LETR.model.utils.matcher import build_matcher
from helper.misc import (
    get_world_size,
    is_dist_avail_and_initialized,
)


def get_loss_weight_dict(cfg):
    losses = []
    weight_dict = {}

    for primitive in cfg.names:
        losses.append(f"{primitive}_labels")
        losses.append(f"loss_{primitive}")
        weight_dict[f"loss_ce_{primitive}"] = 1
        weight_dict[f"loss_{primitive}"] = cfg.line_loss_coef
    aux_layer = cfg.dec_layers

    if cfg.aux_loss:
        aux_weight_dict = {}
        for i in range(aux_layer - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    print(losses, weight_dict)
    return losses, weight_dict


class SetCriterion(nn.Module):
    def __init__(self, criterion_cfg):
        super().__init__()
        self.cfg = criterion_cfg
        self.names = self.cfg.names
        self.matchers = []
        for name in self.cfg.names:
            self.matchers.append(build_matcher(criterion_cfg.matcher, name=name))
        self.losses, self.weight_dict = get_loss_weight_dict(criterion_cfg)  # OK
        self.eos_coef = criterion_cfg.eos_coef
        empty_weight = torch.ones(self.cfg.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        empty_weight[-2] = criterion_cfg.circle_coef
        self.register_buffer("empty_weight", empty_weight)
        try:
            self.cfg.label_loss_params = eval(
                self.cfg.label_loss_params
            )  # Convert the string to dict.
        except:
            pass

    def loss_lines_labels(
        self,
        outputs,
        targets,
        num_items,  # TODO: remove this
        log=False,
        origin_indices=None,
        primitive="shapes",
    ):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_lines]
        """
        assert "pred_logits" in outputs, "pred_logits not in outputs"
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(origin_indices)
        target_classes_o = torch.cat(
            [t[f"{primitive}_labels"][J] for t, (_, J) in zip(targets, origin_indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.cfg.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        if self.cfg.label_loss_func == "cross_entropy":
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        elif self.cfg.label_loss_func == "focal_loss":
            loss_ce = self.label_focal_loss(
                src_logits.transpose(1, 2),
                target_classes,
                self.empty_weight,
                **self.cfg.label_loss_params,
            )
        else:
            raise ValueError(f"loss {self.cfg.label_loss_func} unknown")

        losses = {f"loss_ce_{primitive}": loss_ce}
        return losses

    def label_focal_loss(self, input, target, weight, gamma=2.0):
        """Focal loss for label prediction."""
        # In our case, target has 3 classes: 0 for lines (i.e. line) 1  and 1 for background.
        # The weight here can serve as the alpha hyperparameter in focal loss. However, in focal loss,
        #
        # Ref: https://github.com/facebookresearch/DETR/blob/699bf53f3e3ecd4f000007b8473eda6a08a8bed6/models/segmentation.py#L190
        # Ref: https://medium.com/visionwizard/understanding-focal-loss-a-quick-read-b914422913e7

        # input shape: [batch size, #classes, #queries]
        # target shape: [batch size, #queries]
        # weight shape: [#classes]

        prob = F.softmax(input, 1)  # Shape: [batch size, #classes, #queries].
        ce_loss = F.cross_entropy(
            input, target, weight, reduction="none"
        )  # Shape: [batch size, #queries].
        p_t = prob[:, 1, :] * target + prob[:, 0, :] * (
            1 - target
        )  # Shape: [batch size, #queries]. Note: prob[:,0,:] + prob[:,1,:] should be 1.
        loss = ce_loss * ((1 - p_t) ** gamma)
        loss = (
            loss.mean()
        )  # Original label loss (i.e. cross entropy) does not consider the #lines, so we also do not consider that.
        return loss

    @torch.no_grad()
    def loss_cardinality(
        self, outputs, targets, num_items, origin_indices=None, primitive="shapes"
    ):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty lines
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """  # TODO: log this
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v[f"{primitive}_labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {f"cardinality_error_{primitive}": card_err}
        return losses

    def loss_lines(
        self, outputs, targets, num_items, origin_indices=None, primitive="shapes"
    ):
        assert f"pred_{primitive}" in outputs, "outputs unrecog."

        idx = self._get_src_permutation_idx(origin_indices)

        src_lines = outputs[f"pred_{primitive}"][idx]
        target_lines = torch.cat(
            [t[primitive][i] for t, (_, i) in zip(targets, origin_indices)], dim=0
        )

        loss_line = F.l1_loss(src_lines, target_lines, reduction="none")

        losses = {}
        losses[f"loss_{primitive}"] = loss_line.sum() / num_items

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, num_items, primitive, **kwargs):
        if primitive == "lines":
            loss_map = {
                "POST_lines_labels": self.loss_lines_labels,
                "POST_lines": self.loss_lines,
                "lines_labels": self.loss_lines_labels,
                "cardinality_lines": self.loss_cardinality,
                "loss_lines": self.loss_lines,
            }
        elif primitive == "circles":
            loss_map = {
                "POST_circles_labels": self.loss_lines_labels,
                "POST_circles": self.loss_lines,
                "circles_labels": self.loss_lines_labels,
                "loss_circles": self.loss_lines,
                "cardinality_circles": self.loss_cardinality,
            }
        else:
            loss_map = {
                "POST_shapes_labels": self.loss_lines_labels,
                "POST_shapes": self.loss_lines,
                "shapes_labels": self.loss_lines_labels,
                "loss_shapes": self.loss_lines,
                "cardinality_shapes": self.loss_cardinality,
            }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](
            outputs, targets, num_items, primitive=primitive, **kwargs
        )

    def forward(self, outputs_dict, targets, primitive="shapes", origin_indices=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # for primitive in ["lines", "circles"]:
        # for primitive in primitives:
        losses = {}

        for primitive, matcher in zip(self.names, self.matchers):
            outputs = outputs_dict[primitive]
            outputs_without_aux = {
                k: v for k, v in outputs.items() if k != "aux_outputs"
            }

            origin_indices = matcher(outputs_without_aux, targets)

            num_items = sum(len(t[f"{primitive}_labels"]) for t in targets)

            num_items = torch.as_tensor(
                [num_items],
                dtype=torch.float,
                device=next(iter(outputs.values())).device,
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_items)
            num_items = torch.clamp(num_items / get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.losses:
                if primitive in loss:
                    losses.update(
                        self.get_loss(
                            loss,
                            outputs,
                            targets,
                            num_items,
                            origin_indices=origin_indices,
                            primitive=primitive,  # TODO: replace name with primitive in other functions
                        )
                    )

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            aux_name = "aux_outputs"
            if aux_name in outputs:
                for i, aux_outputs in enumerate(outputs[aux_name]):
                    origin_indices = matcher(aux_outputs, targets)
                    for loss in self.losses:
                        kwargs = {}
                        if "labels" in loss:
                            # Logging is enabled only for the last layer
                            kwargs = {"log": False}
                        if primitive in loss:
                            l_dict = self.get_loss(
                                loss,
                                aux_outputs,
                                targets,
                                num_items,
                                origin_indices=origin_indices,
                                primitive=primitive,
                                **kwargs,
                            )
                            l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                            losses.update(l_dict)
        return losses
