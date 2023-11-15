import torch
from torch import nn
import torch.nn.functional as F


# FIXME: minor, change the name of lines to smth more general
class PostProcess_Line(nn.Module):

    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, output_type="prediction"):
        super(PostProcess_Line, self).__init__()
        self.output_type = output_type

    @torch.no_grad()
    def forward(self, outputs, target_sizes, primitive="shapes"):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        # print("output_type is ", output_type)
        def get_prediction_dict(out_logits, out_line):
            assert len(out_logits) == len(target_sizes), "shape mismatch"
            assert (
                target_sizes.shape[1] == 2
            ), "target_sizes must have shape (batch_size x 2)"

            prob = F.softmax(out_logits, -1)
            scores, labels = prob[..., :-1].max(-1)
            img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            lines = out_line * scale_fct[:, None, :]

            results = [
                {"scores": s, "labels": l, primitive: b}
                for s, l, b in zip(scores, labels, lines)
            ]
            return results

        if self.output_type == "prediction":
            out_logits, out_line = outputs["pred_logits"], outputs[f"pred_{primitive}"]
            results = get_prediction_dict(out_logits, out_line)
        elif self.output_type == "prediction_POST":
            out_logits, out_line = (
                outputs["pred_logits"],
                outputs[f"POST_pred_{primitive}"],
            )
            results = get_prediction_dict(out_logits, out_line)

        elif self.output_type == "ground_truth":
            results = []
            for dic in outputs:
                lines = dic[primitive]
                img_h, img_w = target_sizes.unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                scaled_lines = lines * scale_fct
                results.append(
                    {
                        "labels": dic["labels"],
                        primitive: scaled_lines,
                        "image_id": dic["image_id"],
                    }
                )
        else:
            raise ValueError(f"output_type '{self.output_type}' is invalid")
        return results
