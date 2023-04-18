import logging
from typing import Dict, Optional

import ml_collections
import torch
import torch.nn as nn
from einops import rearrange

from abfold.np import residue_constants
from abfold.utils.loss import AlphaFoldLoss
from abfold.utils.rigid_utils import Rotation, Rigid


def unsupervised_compute_fape(
        pred_frames: Rigid,
        target_frames: Rigid,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        length_scale: float,
        l1_clamp_distance: Optional[float] = None,
        eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = torch.mean(normed_error, dim=(-1, -2))

    return normed_error


def unsupervised_backbone_loss(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        clamp_distance: float = 10.0,
        loss_unit_distance: float = 10.0,
        eps: float = 1e-4,
        **kwargs,
) -> torch.Tensor:
    pred_aff = Rigid.from_tensor_7(student_frames)
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    gt_aff = Rigid.from_tensor_7(teacher_frames)
    gt_aff = Rigid(
        Rotation(rot_mats=gt_aff.get_rots().get_rot_mats(), quats=None),
        gt_aff.get_trans(),
    )

    fape_loss = unsupervised_compute_fape(
        pred_aff,
        gt_aff[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss


def unsupervised_sidechain_loss(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        student_atom_positions: torch.Tensor,
        teacher_atom_positions: torch.Tensor,
        clamp_distance: float = 10.0,
        length_scale: float = 10.0,
        eps: float = 1e-4,
        **kwargs,
) -> torch.Tensor:
    # Steamroll the inputs
    student_frames = student_frames[-1]
    batch_dims = student_frames.shape[:-4]
    student_frames = student_frames.view(*batch_dims, -1, 4, 4)
    student_frames = Rigid.from_tensor_4x4(student_frames)
    teacher_frames = teacher_frames[-1]
    teacher_frames = teacher_frames.view(*batch_dims, -1, 4, 4)
    teacher_frames = Rigid.from_tensor_4x4(teacher_frames)

    student_atom_positions = student_atom_positions[-1]
    student_atom_positions = student_atom_positions.view(*batch_dims, -1, 3)
    teacher_atom_positions = teacher_atom_positions[-1]
    teacher_atom_positions = teacher_atom_positions.view(*batch_dims, -1, 3)

    fape = unsupervised_compute_fape(
        student_frames,
        teacher_frames,
        student_atom_positions,
        teacher_atom_positions,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )

    return fape


def unsupervised_fape_loss(
        student_out: Dict[str, torch.Tensor],
        teacher_out: Dict[str, torch.Tensor],
        config: ml_collections.ConfigDict,
) -> torch.Tensor:
    bb_loss = unsupervised_backbone_loss(
        student_frames=student_out["sm"]["frames"],
        teacher_frames=teacher_out["sm"]["frames"],
        **{**config.backbone},
    )

    sc_loss = unsupervised_sidechain_loss(
        student_frames=student_out["sm"]["sidechain_frames"],
        teacher_frames=teacher_out["sm"]["sidechain_frames"],
        student_atom_positions=student_out["sm"]["positions"],
        teacher_atom_positions=teacher_out["sm"]["positions"],
        **{**config.sidechain},
    )

    loss = config.backbone.weight * bb_loss + config.sidechain.weight * sc_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def unsupervised_chi_loss(
        student_angles_sin_cos: torch.Tensor,
        student_unnormalized_angles_sin_cos: torch.Tensor,
        aatype: torch.Tensor,
        teacher_chi_angles_sin_cos: torch.Tensor,
        chi_weight: float,
        angle_norm_weight: float,
        eps=1e-6,
        **kwargs,
) -> torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)

        Args:
            student_angles_sin_cos:
                [*, N, 7, 2] predicted angles
            student_unnormalized_angles_sin_cos:
                The same angles, but unnormalized
            aatype:
                [*, N] residue indices
            teacher_chi_angles_sin_cos:
                [*, N, 7, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """
    pred_angles = student_angles_sin_cos[..., 3:, :]
    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        residue_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->ik",
        residue_type_one_hot.type(student_angles_sin_cos.dtype),
        student_angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic),
    )

    true_chi = teacher_chi_angles_sin_cos[..., 3:, :]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum(
        (true_chi_shifted - pred_angles) ** 2, dim=-1
    )
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

    # The ol' switcheroo
    sq_chi_error = sq_chi_error.permute(
        *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
    )

    sq_chi_loss = torch.mean(sq_chi_error, dim=(-1, -2, -3))

    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(
        torch.sum(student_unnormalized_angles_sin_cos ** 2, dim=-1) + eps
    )
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(
        *range(len(norm_error.shape))[1:-2], 0, -2, -1
    )
    angle_norm_loss = torch.mean(norm_error, dim=(-1, -2, -3))

    loss = loss + angle_norm_weight * angle_norm_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def unsupervised_bondlen_loss(
        student_coords: torch.Tensor,
        teacher_coords: torch.Tensor,
        offsets=[1, 2],
) -> torch.Tensor:
    student_bb_coords = rearrange(
        student_coords[..., :3, :],
        "b l a d -> b (l a) d",
    )
    teacher_bb_coords = rearrange(
        teacher_coords[..., :3, :],
        "b l a d -> b (l a) d",
    )
    losses = []
    for o in offsets:
        pred_lens = torch.norm(student_bb_coords[..., :-o, :] - student_bb_coords[..., o:, :], dim=-1)
        target_lens = torch.norm(teacher_bb_coords[..., :-o, :] - teacher_bb_coords[..., o:, :], dim=-1)

        losses.append(
            torch.abs(pred_lens - target_lens, ).mean() / o)

    return torch.mean(torch.stack(losses))


class AlphaFoldUnsupervisedLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config, ratio=0.5):
        super(AlphaFoldUnsupervisedLoss, self).__init__()
        self.config = config
        self.ratio = ratio
        self.alphafold_loss = AlphaFoldLoss(config)

    def forward(self, student_out1, teacher_out1, student_out2, gt, _return_breakdown=False):
        loss_fns = {
            "fape": lambda: unsupervised_fape_loss(
                student_out1,
                teacher_out1,
                self.config.fape,
            ),
            "supervised_chi": lambda: unsupervised_chi_loss(
                student_out1["sm"]["angles"],
                student_out1["sm"]["unnormalized_angles"],
                teacher_out1["aatype"],
                teacher_out1["sm"]["angles"],
                **{**self.config.supervised_chi},
            ),
            "bondlen": lambda: unsupervised_bondlen_loss(
                student_out1["final_atom_positions"],
                teacher_out1["final_atom_positions"],
            ),
        }

        cum_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            # print(loss_name)
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if (torch.isnan(loss) or torch.isinf(loss)):
                # for k,v in batch.items():
                #    if(torch.any(torch.isnan(v)) or torch.any(torch.isinf(v))):
                #        logging.warning(f"{k}: is nan")
                # logging.warning(f"{loss_name}: {loss}")
                logging.warning(f"unsupervised {loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            # print(f'unsupervised {loss_name}: {loss}')
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()

        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size
        seq_len = torch.mean(student_out1["seq_length"].float())
        cum_loss = cum_loss * torch.sqrt(seq_len)
        supervised_loss = self.alphafold_loss(student_out2, gt)
        cum_loss = self.ratio * cum_loss + (1 - self.ratio) * supervised_loss


        losses["loss"] = cum_loss.detach().clone()

        if (not _return_breakdown):
            return cum_loss

        return cum_loss, losses
