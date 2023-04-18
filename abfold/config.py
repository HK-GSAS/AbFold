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


import ml_collections as mlc

eps = 1e-6

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

config = mlc.ConfigDict(
    {
        'prep_feat': {
            'attention': {
                'module_params': {
                    'c_q': 384,
                    'c_kv': 384,
                    'c_point': 384,
                    'head_dim': 16,
                    'num_heads': 8,
                    'bias_dim': 128,
                },
            },
            'num_layer': 3,
        },
        'ipa': {
            'module_params': {
                'c_s': 384,
                'c_z': 128,  # 原128
                'c_hidden': 16,
                'no_heads': 12,
                'no_qk_points': 4,
                'no_v_points': 8
            },
            'num_layer': 3,  # 原 3
        },
        'z_conv': {
            'module_params': {
                'c_in': 128,
                'c_out': 128,
            }
        },
        'angle_resnet': {
            'c_in': 384,
            'c_hidden': 128,
            'no_blocks': 1,
            'no_angles': 7,
            'epsilon': 1e-12
        },
        "heads": {
            "lddt": {
                "no_bins": 50,
                "c_in": 384,
                "c_hidden": 128,
            },
            "distogram": {
                "c_z": 128,
                "no_bins": 64,
            },
            "tm": {
                "c_z": 128,
                "no_bins": 64,
                "enabled": False,
            },
            "masked_msa": {
                "c_m": 256,
                "c_out": 23,
            },
            "experimentally_resolved": {
                "c_s": 384,
                "c_out": 37,
            },
        },
        "loss": {
            "fape": {
                "backbone": {
                    "clamp_distance": 10.0,
                    "loss_unit_distance": 10.0,
                    "weight": 0.5,
                },
                "sidechain": {
                    "clamp_distance": 10.0,
                    "length_scale": 10.0,
                    "weight": 0.5,
                },
                "eps": 1e-4,
                "weight": 1.0,
            },
            "supervised_chi": {
                "chi_weight": 0.5,
                "angle_norm_weight": 0.01,
                "eps": eps,  # 1e-6,
                "weight": 1.0,
            },
            "violation": {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "eps": eps,  # 1e-6,
                "weight": 1.0,
            },
            "bondlen": {
                "weight": 1.0,
            },
            "eps": eps,
        },
        "data": {
            "common": {
                "feat": {
                    "aatype": [NUM_RES],
                    "all_atom_mask": [NUM_RES, None],
                    "all_atom_positions": [NUM_RES, None, None],
                    "alt_chi_angles": [NUM_RES, None],
                    "atom14_alt_gt_exists": [NUM_RES, None],
                    "atom14_alt_gt_positions": [NUM_RES, None, None],
                    "atom14_atom_exists": [NUM_RES, None],
                    "atom14_atom_is_ambiguous": [NUM_RES, None],
                    "atom14_gt_exists": [NUM_RES, None],
                    "atom14_gt_positions": [NUM_RES, None, None],
                    "atom37_atom_exists": [NUM_RES, None],
                    "backbone_rigid_mask": [NUM_RES],
                    "backbone_rigid_tensor": [NUM_RES, None, None],
                    "chi_angles_sin_cos": [NUM_RES, None, None],
                    "chi_mask": [NUM_RES, None],
                    "is_distillation": [],
                    "no_recycling_iters": [],
                    "pseudo_beta": [NUM_RES, None],
                    "pseudo_beta_mask": [NUM_RES],
                    "residue_index": [NUM_RES],
                    "residx_atom14_to_atom37": [NUM_RES, None],
                    "residx_atom37_to_atom14": [NUM_RES, None],
                    "resolution": [],
                    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
                    "rigidgroups_group_exists": [NUM_RES, None],
                    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
                    "rigidgroups_gt_exists": [NUM_RES, None],
                    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
                    "seq_length": [],
                    "seq_mask": [NUM_RES],
                    "target_feat": [NUM_RES, None],
                    "use_clamped_fape": [],
                    "s": [NUM_RES, None],
                    "z_self": [NUM_RES, NUM_RES, None],
                    "z_cross": [NUM_RES, NUM_RES, None],
                    "crop_start": [],
                    "point_feat": [NUM_RES, None],
                },
                "max_recycling_iters": 3,
                "msa_cluster_features": True,
                "reduce_msa_clusters_by_max_templates": False,
                "resample_msa_in_recycling": True,
                "template_features": [
                    "template_all_atom_positions",
                    "template_sum_probs",
                    "template_aatype",
                    "template_all_atom_mask",
                ],
                "unsupervised_features": [
                    "aatype",
                    "residue_index",
                    "msa",
                    "num_alignments",
                    "seq_length",
                    "between_segment_residues",
                    "deletion_matrix",
                    "no_recycling_iters",
                ],
                "use_templates": False,
                "use_template_torsion_angles": False,
            },
            "supervised": {
                "clamp_prob": 0.9,
                "supervised_features": [
                    "all_atom_mask",
                    "all_atom_positions",
                    "resolution",
                    "use_clamped_fape",
                    "is_distillation",
                ],
            },
            "train": {
                "fixed_size": True,
                "subsample_templates": True,
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 128,
                "max_extra_msa": 1024,
                "max_template_hits": 4,
                "max_templates": 4,
                "shuffle_top_k_prefiltered": 20,
                "crop": True,
                "crop_size": 96,
                "supervised": True,
                "clamp_prob": 0.9,
                "max_distillation_msa_clusters": 1000,
                "uniform_recycling": True,
                "distillation_prob": 0.75,
            },
        },
    }
)
