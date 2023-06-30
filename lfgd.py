import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GAT, GCN
import torch_sparse

from loguru import logger


class LFGD(nn.Module):

    def __init__(self, args):
        super(LFGD, self).__init__()

        self.enc = nn.Sequential(nn.Linear(args.num_features, args.num_hiddens),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(args.num_hiddens, args.num_hiddens),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(args.num_hiddens, args.rank),
                                 nn.LeakyReLU(0.2))
        self.gcn = GAT(args.num_features, args.num_hiddens,
                       3, args.rank, 0.5, nn.LeakyReLU(0.2))
        # self.gcn = GCN(args.num_features, args.num_hiddens,
        #                1, args.num_hiddens)
        self.enc.apply(self.weights_init)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        self.args = args

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x, adj, normalize=True):
        # x = self.gcn(x, torch.stack(torch.where(adj > 0), dim=0))
        u: torch.Tensor = self.enc(x)
        self.u = u
        # u = F.relu(self.gcn(x,torch.stack(torch.where(adj>0),dim=0)))
        # if normalize:
        #     return torch.sigmoid(u@u.t()).round()
        # else:
        #     return u@u.t()

        # u = u/np.sqrt(self.args.rank)

        return torch.relu(u@u.t())
        row_sum = u.norm(dim=1, keepdim=True).abs()+1e-8
        return torch.relu((u@u.t())/(row_sum@row_sum.t()))

    def step(self, x, adj: torch.Tensor):
        n = x.shape[0]
        adj_d: torch.Tensor = self.forward(x, adj, False)
        assert (adj >= 0).all()
        # loss = F.binary_cross_entropy_with_logits(
        #     adj_d, adj, reduction='none', weight=self.args.bce_weight).mean()
        diff_loss = 0*((adj-adj_d).pow(2)*self.args.bce_weight).mean()
        # density_loss = -0.1 * torch.log(adj_d.sum(-1)+1e-8).mean()
        l1_loss =  0.01/self.args.num_nodes * (adj_d.sum(-1).max())
        # l1_loss = 0.001 * (adj_d.sum()/self.args.num_edges).pow(2)

        lambda_norm = -0.1 * torch.log(self.u.norm(dim=0).sum())

        if self.args.dataset.lower() != 'polblogs':
            smooth_loss = 0
        else:
            smooth_loss = 0

        loss = diff_loss+lambda_norm+l1_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.args.epoch in [0, 199]:
            logger.info(
                f"epoch {self.args.epoch:d}, loss {loss:.4f} = {diff_loss:.4f}+{lambda_norm:.4f}+{l1_loss:.4f}")
            # logger.warning(torch.norm(
            #     self.gcn(x,torch.stack(torch.where(adj>0),dim=0)), dim=0).detach().cpu().numpy())

    def sample(self, x, adj):
        n = self.args.num_nodes
        m = self.args.num_edges
        size = int(m//2)
        degree = adj.sum(-1)+1
        edge_index = torch.stack(torch.where(adj > 0))
        # new selected edges
        node1 = torch.multinomial(
            torch.ones(x.shape[0], device=self.args.device),
            # degree,
            size, replacement=True)
        node2 = torch.multinomial(
            torch.ones(x.shape[0], device=self.args.device),
            # degree.max()-degree+1,
            size, replacement=True)
        ns_edge_index = torch.stack([node1, node2], dim=0)
        ns_edge_index = torch.cat(
            [ns_edge_index, ns_edge_index.flip(0)], dim=1)
        # concatenate
        edge_index = torch.cat([edge_index, ns_edge_index], dim=-1)
        edge_weight = torch.ones_like(edge_index[0], device=self.args.device)
        # collapse
        edge_index, edge_weight = torch_sparse.coalesce(
            edge_index, edge_weight, n, n, op="max")

        return edge_index, edge_weight

    def denoise(self, x, adj):
        n = self.args.num_nodes
        m = self.args.num_edges

        d = self.forward(x, adj).round()


        edge_index, edge_weight = self.sample(x, adj)

        adj = torch.sparse_coo_tensor(edge_index, edge_weight, [
                                      n, n], device=adj.device).to_dense()

        # denoised_adj = torch.zeros_like(d)
        # edge_index = self.sample(x,adj)
        # denoised_adj[edge_index] = d[edge_index]

        # logger.warning(f"denoised_adj.mean: {denoised_adj.mean().item():.3e}")

        # return denoised_adj

        # return d

        return d * adj
