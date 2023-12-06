# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch as th
import numpy as np

import logging

from .vgg import VGGLossMasked

logger = logging.getLogger("dva.{__name__}")


class TotalMVPLoss(nn.Module):
    def __init__(self, weights, assets=None):
        super().__init__()

        self.weights = weights

        if "vgg" in self.weights:
            self.vgg_loss = VGGLossMasked()

    def forward(self, inputs, preds, iteration=None):

        loss_dict = {"loss_total": 0.0}

        B = inputs["image"].shape

        # rgb
        target_rgb = inputs["image"].permute(0, 2, 3, 1)
        # removing the mask
        target_rgb = target_rgb * inputs["image_mask"][:, 0, :, :, np.newaxis]

        rgb = preds["rgb"]
        loss_rgb_mse = th.mean(((rgb - target_rgb) / 16.0) ** 2.0)
        loss_dict.update(loss_rgb_mse=loss_rgb_mse)

        alpha = preds["alpha"]

        # mask loss
        target_mask = inputs["image_mask"][:, 0].to(th.float32)
        loss_mask_mae = th.mean((target_mask - alpha).abs())
        loss_dict.update(loss_mask_mae=loss_mask_mae)

        B = alpha.shape[0]

        # beta prior on opacity
        loss_alpha_prior = th.mean(
            th.log(0.1 + alpha.reshape(B, -1))
            + th.log(0.1 + 1.0 - alpha.reshape(B, -1))
            - -2.20727
        )
        loss_dict.update(loss_alpha_prior=loss_alpha_prior)

        prim_scale = preds["prim_scale"]
        loss_prim_vol_sum = th.mean(th.sum(th.prod(100.0 / prim_scale, dim=-1), dim=-1))
        if th.isnan(loss_prim_vol_sum):
            loss_dict.update(loss_prim_vol_sum = prim_scale)
            loss_total = (
                self.weights.rgb_mse * loss_rgb_mse
                + self.weights.mask_mae * loss_mask_mae
                + self.weights.alpha_prior * loss_alpha_prior
            )
        else:
            loss_total = (
                self.weights.rgb_mse * loss_rgb_mse
                + self.weights.mask_mae * loss_mask_mae
                + self.weights.alpha_prior * loss_alpha_prior
                + self.weights.prim_vol_sum * loss_prim_vol_sum
            )
            loss_dict.update(loss_prim_vol_sum=loss_prim_vol_sum)


        if "embs_l2" in self.weights:
            loss_embs_l2 = th.sum(th.norm(preds["embs"], dim=1))
            loss_total += self.weights.embs_l2 * loss_embs_l2
            loss_dict.update(loss_embs_l2=loss_embs_l2)

        if "vgg" in self.weights:
            loss_vgg = self.vgg_loss(
                rgb.permute(0, 3, 1, 2),
                target_rgb.permute(0, 3, 1, 2),
                inputs["image_mask"],
            )
            loss_total += self.weights.vgg * loss_vgg
            loss_dict.update(loss_vgg=loss_vgg)

        if "prim_scale_var" in self.weights:
            log_prim_scale = th.log(prim_scale)
            # NOTE: should we detach this?
            log_prim_scale_mean = th.mean(log_prim_scale, dim=1, keepdim=True)
            loss_prim_scale_var = th.mean((log_prim_scale - log_prim_scale_mean) ** 2.0)
            loss_total += self.weights.prim_scale_var * loss_prim_scale_var
            loss_dict.update(loss_prim_scale_var=loss_prim_scale_var)

        loss_dict["loss_total"] = loss_total

        return loss_total, loss_dict


def process_losses(loss_dict, reduce=True, detach=True):
    """Preprocess the dict of losses outputs."""
    result = {
        k.replace("loss_", ""): v for k, v in loss_dict.items() if k.startswith("loss_")
    }
    if detach:
        result = {k: v.detach() for k, v in result.items()}
    if reduce:
        result = {k: float(v.mean().item()) for k, v in result.items()}
    return result
