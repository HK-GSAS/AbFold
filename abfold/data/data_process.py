# Copyright 2022 Ahdritz, Gustaf and Bouatta, Nazim and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import os
import pickle
import random
from functools import wraps
from tqdm import tqdm
from typing import Mapping, Tuple, List, Dict, Sequence

import ml_collections
import numpy as np
import torch

from abfold.config import config
from abfold.data import data_transforms
from abfold.np import residue_constants
from abfold.np import protein
from abfold.data import input_pipeline
from abfold.data.align_two_seqs import needle
from abfold.utils.rigid_utils import Rigid, Rotation

FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


def make_data_config(
        config: ml_collections.ConfigDict,
        mode: str,
        num_res: int,
) -> Tuple[ml_collections.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res

    feature_names = cfg.common.unsupervised_features

    if cfg.common.use_templates:
        feature_names += cfg.common.template_features

    if cfg[mode].supervised:
        feature_names += cfg.supervised.supervised_features

    return cfg, feature_names


def np_to_tensor_dict(
        np_example: Mapping[str, np.ndarray],
        features: Sequence[str],
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    tensor_dict = {
        k: torch.tensor(v) for k, v in np_example.items() if k in features
    }

    return tensor_dict


def np_to_tensor_dict_without_filter(
        np_example: Mapping[str, np.ndarray]
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    tensor_dict = {
        k: torch.tensor(v) for k, v in np_example.items()
    }

    return tensor_dict


def tensor_dict_to_np(tensor_dict: Mapping[str, torch.Tensor]):
    np_dict = {
        k: v.numpy() for k, v in tensor_dict.items()
    }

    return np_dict


def nonensembled_transform_fns(common_cfg, mode_cfg):
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_transforms.cast_to_64bit_ints,
        # data_transforms.correct_msa_restypes,
        data_transforms.squeeze_features,
        # data_transforms.randomly_replace_msa_with_unknown(0.0),
        data_transforms.make_seq_mask,
        # data_transforms.make_msa_mask,
        # data_transforms.make_hhblits_profile,
    ]

    transforms.extend(
        [
            data_transforms.make_atom14_masks,
        ]
    )

    if mode_cfg.supervised:
        transforms.extend(
            [
                data_transforms.make_atom14_positions,
                data_transforms.atom37_to_frames,
                data_transforms.atom37_to_torsion_angles(""),
                data_transforms.make_pseudo_beta(""),
                data_transforms.get_backbone_frames,
                data_transforms.get_chi_angles,
            ]
        )

    return transforms


def curry1(f):
    """Supply all arguments but the first."""

    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


@curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def np_example_to_features(
        np_example: FeatureDict,
        config: ml_collections.ConfigDict,
        mode: str,
):
    np_example = dict(np_example)
    num_res = int(np_example["seq_length"][0])
    cfg, feature_names = make_data_config(config, mode=mode, num_res=num_res)

    if "deletion_matrix_int" in np_example:
        np_example["deletion_matrix"] = np_example.pop(
            "deletion_matrix_int"
        ).astype(np.float32)

    tensor_dict = np_to_tensor_dict(
        np_example=np_example, features=feature_names
    )
    with torch.no_grad():
        nonensembled = nonensembled_transform_fns(
            cfg.common,
            cfg[mode],
        )

        features = compose(nonensembled)(tensor_dict)

    return {k: v for k, v in features.items()}


# def np_example_to_features(
#     np_example: FeatureDict,
#     config: ml_collections.ConfigDict,
#     mode: str,
# ):
#     np_example = dict(np_example)
#     num_res = int(np_example["seq_length"][0])
#     cfg, feature_names = make_data_config(config, mode=mode, num_res=num_res)
#
#     if "deletion_matrix_int" in np_example:
#         np_example["deletion_matrix"] = np_example.pop(
#             "deletion_matrix_int"
#         ).astype(np.float32)
#
#     tensor_dict = np_to_tensor_dict(
#         np_example=np_example, features=feature_names
#     )
#     with torch.no_grad():
#         features = input_pipeline.process_tensors_from_config(
#             tensor_dict,
#             cfg.common,
#             cfg[mode],
#         )
#
#     return {k: v for k, v in features.items()}


def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]]
        for i in range(len(aatype))
    ])


def make_sequence_features(
        sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features


def process_pdb(pdb_path, seq, chain_id):
    with open(pdb_path, 'r') as f:
        pdb_str = f.read()

    input_sequence = protein.get_seq_from_pdb_str(pdb_str, chain_id)
    align = needle(seq, input_sequence)
    protein_object = protein.from_pdb_string(pdb_str, chain_id, align, seq)

    # input_sequence_l = protein.get_seq_from_pdb_str(pdb_str, 'L')
    # align_l = needle(seq_l, input_sequence_l)
    # protein_object_l = protein.from_pdb_string(pdb_str, 'L', align_l, seq_l)
    # input_sequence_l = _aatype_to_str_sequence(protein_object_l.aatype)

    pdb_feats = {}
    description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
    pdb_feats.update(
        make_sequence_features(
            sequence=seq,
            description=description,
            num_res=len(seq),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(0.).astype(np.float32)

    return np_example_to_features({**pdb_feats}, config['data'], 'train')


def process_fasta(fasta_path):
    with open(fasta_path) as f:
        lines = f.readlines()
        seq_h = lines[1]
        seq_l = lines[3]

    return seq_h.strip(), seq_l.strip()


def process_repr(repr_path, seq_h):
    with open(repr_path, 'rb') as f:
        data = pickle.load(f)

    s_h, s_l = np.split(data['single'], [len(seq_h)], 0)
    z = data['pair']
    s_h = torch.from_numpy(s_h)
    s_l = torch.from_numpy(s_l)
    z = torch.from_numpy(z)

    return s_h, s_l, z


def process_repr_single_pair(single_repr_path, pair_repr_path, seq_h):
    single = np.load(single_repr_path)
    pair = np.load(pair_repr_path)

    s_h, s_l = np.split(single, [len(seq_h)], 0)
    z = pair
    s_h = torch.from_numpy(s_h)
    s_l = torch.from_numpy(s_l)
    z = torch.from_numpy(z)

    return s_h, s_l, z


# def process_repr(repr_path):
#     with open(repr_path, 'rb') as f:
#         data = pickle.load(f)
#
#     s = data['single']
#     z = data['pair']
#     s = torch.from_numpy(s)
#     z = torch.from_numpy(z)
#
#     return s, z

def process_train_data(train_data_path):
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def process_antibert_feats(antibert_feats_dir):
    with open(antibert_feats_dir, 'rb') as f:
        data = torch.load(f)

    return data[0].squeeze().detach(), data[1].squeeze().detach()


def crop_feat(features, mode='train', ensemble_seed=None):
    if ensemble_seed is None:
        ensemble_seed = torch.Generator().seed()
    num_res = int(features["seq_length"].item())
    cfg, _ = make_data_config(config['data'], mode=mode, num_res=num_res)
    ensembled = input_pipeline.ensembled_transform_fns(
        cfg.common,
        cfg[mode],
        ensemble_seed,
    )

    features = compose(ensembled)(features)

    return {k: v for k, v in features.items()}


def crop_pair_feat(pair, crop_start_1, crop_start_2, crop_size=config['data']['train']['crop_size']):
    pair = pair[crop_start_1:crop_start_1 + crop_size, crop_start_2:crop_start_2 + crop_size]
    dim0, dim1 = pair.shape[:2]
    pair = torch.nn.functional.pad(pair, (0, 0, 0, crop_size - dim1, 0, crop_size - dim0))

    return pair


def concat_labels(label_h, label_l):
    output = {}
    not_crop = ['seq_length', 'resolution', 'is_distillation']
    for key in label_h.keys():
        if key in not_crop:
            output[key] = label_h[key]
        else:
            output[key] = torch.cat([label_h[key], label_l[key]], 0)

    output['seq_length'] = len(output['aatype'])

    return output


def load_label(label_dir):
    labels = []
    for file in tqdm(os.listdir(label_dir)):
        file_path = os.path.join(label_dir, file)
        with open(file_path, 'rb') as f:
            label = pickle.load(f)
        label = (np_to_tensor_dict_without_filter(label[0]), np_to_tensor_dict_without_filter(label[1]))
        labels.append(label)

    return labels


def load_single_label(label_path):
    with open(label_path, 'rb') as f:
        label = pickle.load(f)
    label = (np_to_tensor_dict_without_filter(label[0]), np_to_tensor_dict_without_filter(label[1]))

    return label


def fake_label(aatype):
    label = dict()
    seq_len = aatype.shape[0]
    label['aatype'] = aatype
    label['residue_index'] = torch.arange(1, seq_len + 1)
    label['seq_length'] = seq_len
    label['all_atom_positions'] = torch.zeros([seq_len, 37, 3], dtype=torch.float)
    label['all_atom_mask'] = torch.zeros([seq_len, 37], dtype=torch.float)
    label['resolution'] = torch.Tensor([0.]).squeeze()
    label['is_distillation'] = torch.Tensor([0.]).squeeze()
    label['seq_mask'] = torch.zeros([seq_len], dtype=torch.float)
    label['atom14_atom_exists'] = torch.zeros([seq_len, 14], dtype=torch.float)
    label['residx_atom14_to_atom37'] = torch.zeros([seq_len, 14], dtype=torch.int64)
    label['residx_atom37_to_atom14'] = torch.zeros([seq_len, 37], dtype=torch.int64)
    label['atom37_atom_exists'] = torch.zeros([seq_len, 37], dtype=torch.float)
    label['atom14_gt_exists'] = torch.zeros([seq_len, 14], dtype=torch.float)
    label['atom14_gt_positions'] = torch.zeros([seq_len, 14, 3], dtype=torch.float)
    label['atom14_alt_gt_positions'] = torch.zeros([seq_len, 14, 3], dtype=torch.float)
    label['atom14_alt_gt_exists'] = torch.zeros([seq_len, 14], dtype=torch.float)
    label['atom14_atom_is_ambiguous'] = torch.zeros([seq_len, 14], dtype=torch.float)
    label['rigidgroups_gt_frames'] = torch.zeros([seq_len, 8, 4, 4], dtype=torch.float)
    label['rigidgroups_gt_exists'] = torch.zeros([seq_len, 8], dtype=torch.float)
    label['rigidgroups_group_exists'] = torch.zeros([seq_len, 8], dtype=torch.float)
    label['rigidgroups_group_is_ambiguous'] = torch.zeros([seq_len, 8], dtype=torch.float)
    label['rigidgroups_alt_gt_frames'] = torch.zeros([seq_len, 8, 4, 4], dtype=torch.float)
    label['pseudo_beta'] = torch.zeros([seq_len, 3], dtype=torch.float)
    label['pseudo_beta_mask'] = torch.zeros([seq_len], dtype=torch.float)
    label['backbone_rigid_tensor'] = torch.zeros([seq_len, 4, 4], dtype=torch.float)
    label['backbone_rigid_mask'] = torch.zeros([seq_len], dtype=torch.float)
    label['chi_angles_sin_cos'] = torch.zeros([seq_len, 4, 2], dtype=torch.float)
    label['chi_mask'] = torch.zeros([seq_len, 4], dtype=torch.float)
    return label


def get_struct_from_pdb(path, mask=None):
    struct = {'H': {'atom_positions': [], 'aatype': [], 'b_factors': []},
              'L': {'atom_positions': [], 'aatype': [], 'b_factors': []}}
    with open(path) as f:
        lines = f.readlines()
        atom_cnt = 0
        res_idx = 0
        for line in lines:
            if not line or not line.startswith('ATOM'):
                continue
            line = line.split()
            if res_idx != line[5]:
                struct[line[4]]['atom_positions'].append(line[6:9])
                struct[line[4]]['aatype'].append(line[3])
                struct[line[4]]['b_factors'].append(0.0 if float(line[10]) < 1 else 1.0)
                atom_cnt = 1
                res_idx = line[5]
            elif atom_cnt < 3:
                struct[line[4]]['atom_positions'].append(line[6:9])
                atom_cnt += 1

        # deal chain H atom positions
        positions = [[] for _ in range(len(struct['H']['b_factors']))]
        for i, position in enumerate(struct['H']['atom_positions']):
            positions[i // 3].append(position)

        struct['H']['atom_positions'] = np.array(positions, dtype=np.float)

        # deal chain L atom positions
        positions = [[] for _ in range(len(struct['L']['b_factors']))]
        for i, position in enumerate(struct['L']['atom_positions']):
            positions[i // 3].append(position)

        struct['L']['atom_positions'] = np.array(positions, dtype=np.float)

    return struct


# def get_struct_from_pdb(path, mask=None):
#     struct = {'H': {'atom_positions': [], 'aatype': [], 'b_factors': []},
#               'L': {'atom_positions': [], 'aatype': [], 'b_factors': []}}
#     with open(path) as f:
#         lines = f.readlines()
#         atom_cnt = 0
#         res_idx = 0
#         for line in lines:
#             if not line or not line.startswith('ATOM'):
#                 continue
#             line = line.split()
#             if res_idx != line[5]:
#                 struct[line[4]]['aatype'].append(line[3])
#                 if len(struct[line[4]]['aatype']) in mask[line[4]]:
#                     struct[line[4]]['atom_positions'].append([0.0, 0.0, 0.0])
#                     struct[line[4]]['b_factors'].append(1.0)
#                 else:
#                     struct[line[4]]['atom_positions'].append(line[6:9])
#                     struct[line[4]]['b_factors'].append(0.0)
#                 atom_cnt = 1
#                 res_idx = line[5]
#             elif atom_cnt < 3:
#                 if len(struct[line[4]]['aatype']) in mask[line[4]]:
#                     struct[line[4]]['atom_positions'].append([0.0, 0.0, 0.0])
#                 else:
#                     struct[line[4]]['atom_positions'].append(line[6:9])
#                 atom_cnt += 1
#
#         # deal chain H atom positions
#         positions = [[] for _ in range(len(struct['H']['b_factors']))]
#         for i, position in enumerate(struct['H']['atom_positions']):
#             positions[i // 3].append(position)
#
#         struct['H']['atom_positions'] = np.array(positions, dtype=np.float)
#
#         # deal chain L atom positions
#         positions = [[] for _ in range(len(struct['L']['b_factors']))]
#         for i, position in enumerate(struct['L']['atom_positions']):
#             positions[i // 3].append(position)
#
#         struct['L']['atom_positions'] = np.array(positions, dtype=np.float)
#
#     return struct


# def get_struct_from_pdb(path, mask=None, mask_ratio=0.1):
#     struct = {'H': {'atom_positions': [], 'aatype': [], 'b_factors': []},
#               'L': {'atom_positions': [], 'aatype': [], 'b_factors': []}}
#     with open(path) as f:
#         lines = f.readlines()
#         atom_cnt = 0
#         res_idx = 0
#         coord_type = 'igfold'
#         for line in lines:
#             if not line or not line.startswith('ATOM'):
#                 continue
#             line = line.split()
#             if res_idx != line[5]:
#                 coord_type = get_mask(mask_ratio)
#                 struct[line[4]]['aatype'].append(line[3])
#                 struct[line[4]]['b_factors'].append(1.0)
#                 atom_cnt = 1
#                 res_idx = line[5]
#             elif atom_cnt < 3:
#                 atom_cnt += 1
#             else:
#                 continue
#
#             if coord_type == 'igfold':
#                 struct[line[4]]['atom_positions'].append(line[6:9])
#             elif coord_type == 'random':
#                 struct[line[4]]['atom_positions'].append([
#                     random.uniform(0, 100),
#                     random.uniform(0, 100),
#                     random.uniform(0, 100),
#                 ])
#             else:
#                 struct[line[4]]['atom_positions'].append([0.0, 0.0, 0.0])
#
#         # deal chain H atom positions
#         positions = [[] for _ in range(len(struct['H']['b_factors']))]
#         for i, position in enumerate(struct['H']['atom_positions']):
#             positions[i // 3].append(position)
#
#         struct['H']['atom_positions'] = np.array(positions, dtype=np.float)
#
#         # deal chain L atom positions
#         positions = [[] for _ in range(len(struct['L']['b_factors']))]
#         for i, position in enumerate(struct['L']['atom_positions']):
#             positions[i // 3].append(position)
#
#         struct['L']['atom_positions'] = np.array(positions, dtype=np.float)
#
#     return struct


def get_frames_from_pdb(path, mask=None):
    if mask is None:
        mask = {'H': [], 'L': []}
    struct = get_struct_from_pdb(path, mask)
    # h_frames = Rigid.from_3_points(p_neg_x_axis=torch.from_numpy(struct['H']['atom_positions'][..., 0, :]),
    #                                origin=torch.from_numpy(struct['H']['atom_positions'][..., 1, :]),
    #                                p_xy_plane=torch.from_numpy(struct['H']['atom_positions'][..., 2, :]))
    # l_frames = Rigid.from_3_points(p_neg_x_axis=torch.from_numpy(struct['L']['atom_positions'][..., 0, :]),
    #                                origin=torch.from_numpy(struct['L']['atom_positions'][..., 1, :]),
    #                                p_xy_plane=torch.from_numpy(struct['L']['atom_positions'][..., 2, :]))
    h_frames = Rigid.from_3_points(p_neg_x_axis=torch.from_numpy(struct['H']['atom_positions'][..., 2, :]),
                                   origin=torch.from_numpy(struct['H']['atom_positions'][..., 1, :]),
                                   p_xy_plane=torch.from_numpy(struct['H']['atom_positions'][..., 0, :]))
    l_frames = Rigid.from_3_points(p_neg_x_axis=torch.from_numpy(struct['L']['atom_positions'][..., 2, :]),
                                   origin=torch.from_numpy(struct['L']['atom_positions'][..., 1, :]),
                                   p_xy_plane=torch.from_numpy(struct['L']['atom_positions'][..., 0, :]))
    return h_frames.to_tensor_4x4(), l_frames.to_tensor_4x4(), \
           torch.tensor(struct['H']['b_factors']), torch.tensor(struct['L']['b_factors'])


def load_mask_for_coord(path, keys=None):
    if keys is None:
        keys = ['H_cdr3']

    id_to_mask = dict()
    with open(path, 'rb') as f:
        datas = pickle.load(f)

    for pdb_id, val in datas.items():
        h_mask = []
        l_mask = []
        for key in keys:
            chain, cdr = key.split('_')
            beg, end = datas[pdb_id][chain][cdr].split(':')
            beg, end = int(beg), int(end)
            if chain == 'H':
                h_mask.extend(list(range(beg, end)))
            else:
                l_mask.extend(list(range(beg, end)))

        id_to_mask[pdb_id] = {'H': h_mask, 'L': l_mask}

    return id_to_mask


def get_mask(ratio=0.1):
    """
    按一定比例进行mask, mask时按1:1:8的比例返回正确信息、随机信息、未知信息
    :param ratio: mask ratio
    :return:
    """
    mask = random.choices([0, 1], cum_weights=[ratio, 1], k=1)[0]
    if mask:
        return 'igfold'
    else:
        info = random.choices([0, 1, 2], weights=[1, 1, 8], k=1)[0]
        if info == 0:
            return 'igfold'
        elif info == 1:
            return 'random'
        else:
            return 'origin'
