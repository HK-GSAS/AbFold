import os

import torch
from torch.utils.data import Dataset

from abfold.data.data_process import process_fasta, load_single_label, process_repr, crop_feat, concat_labels, \
    load_label, crop_pair_feat
from abfold.np.residue_constants import str_sequence_to_aatype


class AbFoldDataset(Dataset):
    def __init__(self, label_dir, point_feat_dir):
        super(AbFoldDataset, self).__init__()

        self.label_dir = label_dir
        self.point_feat_dir = point_feat_dir
        self._load_label_ids()
        self.seed = torch.Generator().seed()

    def _load_labels(self):
        self._uncrop_labels = load_label(self.label_dir)

    def _load_label_ids(self):
        self._label_ids = []
        for file in os.listdir(self.label_dir):
            self._label_ids.append(file)

    def __len__(self):
        return len(self._label_ids)

    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self._label_ids[idx])
        label_h, label_l = load_single_label(label_path)

        point_feat_path = os.path.join(self.point_feat_dir, f'{self._label_ids[idx].split(".")[0]}.pt')
        point_feat = torch.load(point_feat_path, map_location=torch.device('cpu')).squeeze()
        if label_h['aatype'].shape[0] + label_l['aatype'].shape[0] != point_feat.shape[0]:
            point_feat = point_feat[:label_h['aatype'].shape[0] + label_l['aatype'].shape[0]]
        point_feat_h, point_feat_l = torch.split(point_feat, [label_h['aatype'].shape[0], label_l['aatype'].shape[0]])
        label_h['point_feat'] = point_feat_h
        label_l['point_feat'] = point_feat_l

        z_hl, z_lh = label_h.pop('z_cross'), label_l.pop('z_cross')
        label_h = crop_feat(label_h, ensemble_seed=self.seed)
        label_l = crop_feat(label_l, ensemble_seed=self.seed)
        s_h, s_l = label_h.pop('s'), label_l.pop('s')
        point_feat_h, point_feat_l = label_h.pop('point_feat'), label_l.pop('point_feat')
        point_feat = torch.cat([point_feat_h, point_feat_l])
        z_hh, z_ll = label_h.pop('z_self'), label_l.pop('z_self')
        h_crop_start, l_crop_start = label_h.pop('crop_start'), label_l.pop('crop_start')
        z_hl = crop_pair_feat(z_hl, h_crop_start, l_crop_start)
        z_lh = crop_pair_feat(z_lh, l_crop_start, h_crop_start)
        aatype_h, aatype_l = label_h['aatype'], label_l['aatype']

        feature = {'s_h': s_h, 's_l': s_l, 'z_hh': z_hh, 'z_hl': z_hl,
                   'z_lh': z_lh, 'z_ll': z_ll, 'aatype_h': aatype_h,
                   'aatype_l': aatype_l, 'point_feat': point_feat}

        label = concat_labels(label_h, label_l)

        return feature, label


class UnsupervisedDataset(Dataset):
    def __init__(self, repr_dir, fasta_dir, point_feat_dir):
        super(UnsupervisedDataset, self).__init__()

        self.repr_dir = repr_dir
        self.fasta_dir = fasta_dir
        self.point_feat_dir = point_feat_dir

        self._repr_ids = []
        for line in os.listdir(repr_dir):
            repr_id = line.split('.')[0]
            self._repr_ids.append(repr_id)

    def __len__(self):
        return len(self._repr_ids)

    def __getitem__(self, idx):
        repr_id = self._repr_ids[idx]
        repr_path = os.path.join(self.repr_dir, f'{repr_id}.pkl')
        fasta_path = os.path.join(self.fasta_dir, f'{repr_id}.fasta')
        file_prefix = str(int(repr_id[2:]))
        point_feat_path = os.path.join(self.point_feat_dir, f'{file_prefix}.pt')

        seq_h, seq_l = process_fasta(fasta_path)
        s_h, s_l, z = process_repr(repr_path, seq_h)
        point_feat = torch.load(point_feat_path, map_location=torch.device('cpu')).squeeze()
        if len(seq_h) + len(seq_l) != point_feat.shape[0]:
            point_feat = point_feat[:len(seq_h) + len(seq_l)]
        point_feat_h, point_feat_l = torch.split(point_feat, [len(seq_h), len(seq_l)])

        z_h, z_l = z.split([s_h.shape[-2], s_l.shape[-2]], -3)
        z_hh, z_hl = z_h.split([s_h.shape[-2], s_l.shape[-2]], -2)
        z_lh, z_ll = z_l.split([s_h.shape[-2], s_l.shape[-2]], -2)
        label_h = dict()
        label_l = dict()
        label_h['s'] = s_h
        label_l['s'] = s_l
        label_h['z_self'] = z_hh
        label_l['z_self'] = z_ll
        label_h['z_cross'] = z_hl
        label_l['z_cross'] = z_lh
        label_h['seq_length'] = torch.IntTensor([s_h.shape[0]]).squeeze()
        label_l['seq_length'] = torch.IntTensor([s_l.shape[0]]).squeeze()
        label_h['aatype'] = torch.tensor(str_sequence_to_aatype(seq_h))
        label_l['aatype'] = torch.tensor(str_sequence_to_aatype(seq_l))
        label_h['point_feat'] = point_feat_h
        label_l['point_feat'] = point_feat_l

        label_h = crop_feat(label_h)
        label_l = crop_feat(label_l)
        s_h = label_h.pop('s')
        s_l = label_l.pop('s')
        z_hh, z_ll = label_h.pop('z_self'), label_l.pop('z_self')
        z_hl, z_lh = label_h.pop('z_cross'), label_l.pop('z_cross')
        h_crop_start, l_crop_start = label_h.pop('crop_start'), label_l.pop('crop_start')
        z_hl = crop_pair_feat(z_hl, h_crop_start, l_crop_start)
        z_lh = crop_pair_feat(z_lh, l_crop_start, h_crop_start)
        aatype_h, aatype_l = label_h['aatype'], label_l['aatype']
        point_feat_h, point_feat_l = label_h.pop('point_feat'), label_l.pop('point_feat')
        point_feat = torch.cat([point_feat_h, point_feat_l])

        feature = {'s_h': s_h, 's_l': s_l, 'z_hh': z_hh, 'z_hl': z_hl,
                   'z_lh': z_lh, 'z_ll': z_ll, 'aatype_h': aatype_h,
                   'aatype_l': aatype_l, 'point_feat': point_feat}

        return feature
