import torch
import torch.nn as nn


class GlobalPredict(nn.Module):
    def __init__(self, c_s, max_len=256):
        super(GlobalPredict, self).__init__()

        self.max_len = max_len

        self.linear1_h = nn.Linear(c_s, 7)
        self.linear1_l = nn.Linear(c_s, 7)
        self.linear2 = nn.Linear(max_len, 7)

    def forward(self, s_h, s_l):
        c_h = self.linear1_h(s_h)
        c_l = self.linear1_l(s_l)

        c = torch.cat((c_h, c_l), -2)
        len_c = c.shape[-1]
        t = torch.sum(c, -2) / len_c

        return t
