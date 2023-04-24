import torch.nn as nn


class P2PRegressor(nn.Module):
    def __init__(self, args):
        super(P2PRegressor, self).__init__()

        input_dim = 3 + (args.feat_dim * (args.block_num + 2))
        self.mlp_0 = nn.Conv1d(input_dim, args.feat_dim*2, 1)
        self.mlp_1 = nn.Conv1d(args.feat_dim*2, args.feat_dim, 1)
        self.mlp_2 = nn.Conv1d(args.feat_dim, args.feat_dim//2, 1)
        self.mlp_3 = nn.Conv1d(args.feat_dim//2, 1, 1)
        self.actvn = nn.ReLU()

    def forward(self, feats):
        # input: (b, c, n)

        output = self.actvn(self.mlp_0(feats))
        output = self.actvn(self.mlp_1(output))
        output = self.actvn(self.mlp_2(output))
        output = self.actvn(self.mlp_3(output))
        # (b, 1, n)
        return output
