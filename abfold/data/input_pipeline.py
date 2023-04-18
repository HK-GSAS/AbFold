# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import torch

from abfold.data import data_transforms


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


def ensembled_transform_fns(common_cfg, mode_cfg, ensemble_seed):
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []

    crop_feats = dict(common_cfg.feat)

    transforms.append(data_transforms.select_feat(list(crop_feats)))
    transforms.append(
        data_transforms.random_crop_to_size(
            mode_cfg.crop_size,
            mode_cfg.max_templates,
            crop_feats,
            mode_cfg.subsample_templates,
            seed=ensemble_seed + 1,
        )
    )
    transforms.append(
        data_transforms.make_fixed_size(
            crop_feats,
            mode_cfg.crop_size,
        )
    )

    return transforms


def process_tensors_from_config(tensors, common_cfg, mode_cfg):
    """Based on the config, apply filters and transformations to the data."""

    ensemble_seed = torch.Generator().seed()

    nonensembled = nonensembled_transform_fns(
        common_cfg,
        mode_cfg,
    )

    tensors = compose(nonensembled)(tensors)

    ensembled = ensembled_transform_fns(
        common_cfg,
        mode_cfg,
        ensemble_seed,
    )

    tensors = compose(ensembled)(tensors)

    return tensors


@data_transforms.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack(
            [dict_i[feat] for dict_i in ensembles], dim=-1
        )
    return ensembled_dict
