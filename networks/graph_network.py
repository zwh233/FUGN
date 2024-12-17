import functools
import torch.nn as nn
import networks.blocks as bs
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from graphmethods import dense_to_sparse
##### Depth-guided enhancement network  #####
##### input:256*256*3, 256*256*1|output:256*256*3

class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout=0.4):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_c, hid_c)
        self.conv2 = GCNConv(hid_c, out_c)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class graph_net(nn.Module):
    def __init__(self, block_size=16, in_c=3, out_c=3, hid_size=1024, bc=64):
        super(graph_net, self).__init__()
        self.conv_ini = nn.Sequential(nn.Conv2d(in_c, bc, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(bc, bc, 3, 1, 1, bias=True))
        self.rb1 = bs.RB()

        # 编码器部分
        self.down_conv = nn.Conv2d(bc, hid_size, kernel_size=block_size, stride=block_size, padding=0)

        # 图卷积层
        self.GCN = GCN(hid_size, 1500, hid_size)

        # 解码器部分
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(hid_size, hid_size//2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(hid_size//2, hid_size//4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(hid_size//4, hid_size//8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose2d(hid_size//8, hid_size//16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, True)
        )

        # 最终调整通道数到输出通道
        self.final_conv = nn.Conv2d(hid_size//16, out_c, 3, 1, 1, bias=True)

    def forward(self, x, adj):
        # 初始卷积
        e1 = self.conv_ini(x)
        e1 = self.rb1(e1)

        # 下采样并通过GCN处理
        d1 = self.down_conv(e1)
        e2 = d1.permute(0, 2, 3, 1).reshape(-1, d1.shape[1])
        e2 = self.GCN(e2, dense_to_sparse(adj).to(x.device))
        e2 = e2.view(x.shape[0], -1, d1.shape[2], d1.shape[3])
        e2 = e2 + d1

        e3 = self.up_conv1(e2)
        e3 = self.up_conv2(e3)
        e3 = self.up_conv3(e3)
        e3 = self.up_conv4(e3) + e1

        # 最终通道数调整
        out = self.final_conv(e3)

        return out