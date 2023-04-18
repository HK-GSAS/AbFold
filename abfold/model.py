import torch
import torch.nn as nn

from abfold.angle_resnet import AngleResnet
from abfold.attention import BiasAttentionModule
from abfold.backbone_update import BackboneUpdate
from abfold.data import data_transforms
from abfold.global_predict import GlobalPredict
from abfold.ipa import InvariantPointAttention
from abfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from abfold.transition import LayerNorm, StructureModuleTransition
from abfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
    atom14_to_atom37,
)
from abfold.utils.rigid_utils import Rigid
from abfold.utils.tensor_utils import (
    dict_multimap,
)


class AbFold(nn.Module):
    """
    abfold
    implemented the abfold mentioned in 2022.9.19
    """

    def __init__(self, config, dropout_rate=0.1, no_transition_layers=1):
        super(AbFold, self).__init__()

        self.dropout_rate = dropout_rate
        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None

        # self_attention + cross_attention
        self.atten_layers = nn.ModuleList([])
        for _ in range(config['prep_feat']['num_layer']):
            atten_module = BiasAttentionModule(**config['prep_feat']['attention']['module_params'])
            self.atten_layers.append(nn.ModuleList([atten_module]))

        self.ipa_layers = nn.ModuleList([])
        for _ in range(config['ipa']['num_layer']):
            h_self_ipa = InvariantPointAttention(**config['ipa']['module_params'])
            l_self_ipa = InvariantPointAttention(**config['ipa']['module_params'])
            h_cross_ipa = InvariantPointAttention(**config['ipa']['module_params'])
            l_cross_ipa = InvariantPointAttention(**config['ipa']['module_params'])

            ipa_dropout = nn.Dropout(self.dropout_rate)
            layer_norm_ipa = LayerNorm(config['ipa']['module_params']['c_s'])

            h_transition = StructureModuleTransition(
                config['ipa']['module_params']['c_s'],
                no_transition_layers,
                self.dropout_rate,
            )

            l_transition = StructureModuleTransition(
                config['ipa']['module_params']['c_s'],
                no_transition_layers,
                self.dropout_rate,
            )

            bb_update = BackboneUpdate(config['ipa']['module_params']['c_s'])
            self.ipa_layers.append(nn.ModuleList([h_self_ipa, l_self_ipa, h_cross_ipa, l_cross_ipa,
                                                  ipa_dropout, layer_norm_ipa, h_transition, l_transition, bb_update]))

        self.angle_resnet = AngleResnet(**config['angle_resnet'])
        self.global_predict = GlobalPredict(config['ipa']['module_params']['c_s'])

        self.val_linear = nn.Linear(7, 7)

    def forward(
            self,
            batch,
            mask_h=None,
            mask_l=None,
    ):
        """
        :param s_h:
            [*, N_res_h, C_s] single representation
        :param s_l:
            [*, N_res_l, C_s] single representation
        :param z:
            [*, N_res_h + N_res_l, N_res_h + N_res_l, C_z] pair representation
        :param aatype_h:
            [*, N_res_h]
        :param aatype_l:
            [*, N_res_l]
        :param r_h:
            [*, N_res_h]
        :param r_l:
            [*, N_res_l]
        :param mask_h:
        :param mask_l:
        :return:
        """
        s_h = batch['s_h']
        s_l = batch['s_l']
        z_hh, z_hl = batch['z_hh'], batch['z_hl']
        z_lh, z_ll = batch['z_lh'], batch['z_ll']
        aatype_h = batch['aatype_h']
        aatype_l = batch['aatype_l']
        r_h, r_l = batch.get('r_h', None), batch.get('r_l', None)
        point_feat = batch['point_feat']

        if r_h is None:
            r_h = Rigid.identity(
                s_h.shape[:-1],
                s_h.dtype,
                s_h.device,
                True,
                fmt="quat",
            )
        else:
            r_h = Rigid.from_tensor_4x4(r_h)
        if r_l is None:
            r_l = Rigid.identity(
                s_l.shape[:-1],
                s_l.dtype,
                s_l.device,
                True,
                fmt="quat",
            )
        else:
            r_l = Rigid.from_tensor_4x4(r_l)
        if mask_h is None:
            mask_h = s_h.new_ones(s_h.shape[:-1])
        if mask_l is None:
            mask_l = s_l.new_ones(s_l.shape[:-1])

        s_h_initial = s_h
        s_l_initial = s_l

        rigids = []
        rigids.append([r_h, r_l, s_h, s_l])

        # self_attention and cross_attention
        for atten_module in self.atten_layers:
            s_h, s_l = atten_module[0](s_h, s_l, z_hh, z_hl, z_ll, z_lh, point_feat)

        # self_ipa and cross_ipa
        for h_self_ipa, l_self_ipa, h_cross_ipa, l_cross_ipa, ipa_dropout, \
            layer_norm_ipa, h_transition, l_transition, bb_update in self.ipa_layers:
            s_h = s_h + h_self_ipa(s_h, s_h, z_hh, r_h, r_h, mask_h, mask_h)
            s_l = s_l + l_self_ipa(s_l, s_l, z_ll, r_l, r_l, mask_l, mask_l)
            s_l = s_l + l_cross_ipa(s_l, s_h, z_lh, r_l, r_h, mask_l, mask_h)
            s_h = s_h + h_cross_ipa(s_h, s_l, z_hl, r_h, r_l, mask_h, mask_l)

            s_h = layer_norm_ipa(ipa_dropout(s_h))
            s_l = layer_norm_ipa(ipa_dropout(s_l))
            s_h = h_transition(s_h)
            s_l = l_transition(s_l)

            r_h = r_h.compose_q_update_vec(bb_update(s_h))
            r_l = r_l.compose_q_update_vec(bb_update(s_l))
            rigids.append([r_h, r_l, s_h, s_l])

        r_global = Rigid.from_tensor_7(self.global_predict(s_h, s_l)[..., None, :], True)
        r_h = r_h.scale_translation(20)
        r_l = r_l.scale_translation(20)

        # [*, N, 7, 2]
        unnormalized_angles_h, angles_h = self.angle_resnet(s_h, s_h_initial)
        unnormalized_angles_l, angles_l = self.angle_resnet(s_l, s_l_initial)

        all_frames_to_global_h = self.torsion_angles_to_frames(
            r_h,
            angles_h,
            aatype_h,
        )

        all_frames_to_global_l = self.torsion_angles_to_frames(
            r_l,
            angles_l,
            aatype_l,
        )

        pred_xyz_h = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global_h,
            aatype_h,
        )

        pred_xyz_l = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global_l,
            aatype_l,
        )

        preds = {
            "frames": torch.cat((r_h.to_tensor_7(), r_l.to_tensor_7()), -2),
            "sidechain_frames": torch.cat((all_frames_to_global_h.to_tensor_4x4(),
                                           all_frames_to_global_l.to_tensor_4x4()), -4),
            "unnormalized_angles": torch.cat((unnormalized_angles_h, unnormalized_angles_l), -3),
            "angles": torch.cat((angles_h, angles_l), -3),
            "positions": torch.cat((pred_xyz_h, pred_xyz_l), -3),
            "t_global_tensor_7": r_global.to_tensor_7(),
        }
        outputs_sm = [preds]
        outputs_sm = dict_multimap(torch.stack, outputs_sm)
        outputs_sm["single"] = torch.cat([s_h, s_l], -2)

        outputs = {'sm': outputs_sm}
        aatype = torch.cat([aatype_h, aatype_l], -1)
        feats = self._generate_feats_from_aatype(aatype)
        outputs["aatype"] = aatype
        outputs["seq_length"] = torch.full([aatype.shape[0]], fill_value=aatype.shape[1], device=aatype.device)
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        r_h.stop_rot_gradient()
        r_l.stop_rot_gradient()
        r_global.stop_rot_gradient()

        return outputs

    def _generate_feats_from_aatype(self, aatype):
        return data_transforms.make_atom14_masks({"aatype": aatype})

    def _init_residue_constants(self, float_dtype, device):
        if self.default_frames is None:
            self.default_frames = torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None:
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None:
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None:
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
            self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

    def get_positions_from_r(self, r_h, r_l, s_h, s_l, s_h_initial, s_l_initial, aatype_h, aatype_l):
        unnormalized_angles_h, angles_h = self.angle_resnet(s_h, s_h_initial)
        unnormalized_angles_l, angles_l = self.angle_resnet(s_l, s_l_initial)

        all_frames_to_global_h = self.torsion_angles_to_frames(
            r_h,
            angles_h,
            aatype_h,
        )

        all_frames_to_global_l = self.torsion_angles_to_frames(
            r_l,
            angles_l,
            aatype_l,
        )

        pred_xyz_h = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global_h,
            aatype_h,
        )

        pred_xyz_l = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global_l,
            aatype_l,
        )

        positions = torch.cat((pred_xyz_h, pred_xyz_l), -3)
        aatype = torch.cat([aatype_h, aatype_l], -1)
        feats = self._generate_feats_from_aatype(aatype)
        positions = atom14_to_atom37(positions, feats)

        return positions
