from matplotlib.ticker import LinearLocator
from matplotlib import cm
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
import scipy.sparse as ssp
import numpy as np
import torch
from lfgd import LFGD
from deeprobust.graph.defense import GCN, GAT, GCNSVD, DeepWalk, Node2Vec
from deeprobust.graph.data import Dataset, PrePtbDataset, Dpr2Pyg
import argparse
from loguru import logger
import sys
import random

# logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--fastmode', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--dataset', type=str, default='polblogs')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--rank', type=int, default=30)
parser.add_argument('--outfile', type=str, default='mfgc')
parser.add_argument('--attack', type=str, default='meta')
parser.add_argument('--ptb_rate', type=str, default=0.2)
args = parser.parse_args()
args.device = device = torch.device(
    f'cuda:{args.cuda_id:d}' if torch.cuda.is_available() else 'cpu')

#
# ,'ACM']:
for args.dataset in ['Cora', 'CiteSeer', 'Cora_ML', 'PolBlogs', 'ACM', 'PubMed']:
    # for args.dataset in ['Cora','CiteSeer','PolBlogs']:
    # for args.dataset in ['PubMed']:
# for args.dataset in ['PolBlogs']:
    args.dataset = args.dataset.lower()
    for args.ptb_rate in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        # for args.ptb_rate in [0]:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True

        data = Dataset(root='./datasets/',
                       name=args.dataset.lower(), seed=15)
        args.data = data
        adj, x, labels = data.adj, data.features, data.labels

        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        if args.ptb_rate != 0:

            if args.dataset != 'acm':
                adj = PrePtbDataset(root='./datasets/',
                                    name=args.dataset.lower(),
                                    attack_method='meta',
                                    ptb_rate=args.ptb_rate).adj
            else:
                adj = ssp.load_npz(
                    f'./datasets/{args.dataset.lower()}_{args.attack}_adj_{args.ptb_rate:g}.npz')

        x = torch.FloatTensor(x.todense()).to(device)
        x = x / x.sum(-1, keepdim=True)
        adj = torch.FloatTensor(adj.todense()).to(device)
        if args.dataset.lower() == "pubmed":
            adj = torch.relu(
                adj - 1000000*torch.eye(adj.shape[0], device=device))
        labels = torch.LongTensor(labels).to(device)
        idx_train = torch.LongTensor(idx_train).to(device)
        idx_val = torch.LongTensor(idx_val).to(device)
        idx_test = torch.LongTensor(idx_test).to(device)

        # for i in torch.linalg.svdvals(adj):
        #     print(f"{i.item():.2e}",end=', ')
        # print('')

        args.num_nodes = x.shape[0]
        args.num_features = x.shape[1]
        args.num_classes = int(max(labels) + 1)
        args.num_hiddens = 100
        args.num_edges = adj.sum().item()
        # args.rank = args.num_classes
        args.bce_weight = adj + (1-2*adj)*adj.mean()
        args.bce_weight /= args.bce_weight.min()

        import time
        begin_time = time.time()

        model = LFGD(args).to(device)

        for args.epoch in range(args.epochs):
            model.step(x, adj)

        denoised_adj = model.denoise(x, adj).detach()

        logger.info(adj.sum(-1).unique(return_counts=True))
        logger.info(denoised_adj.sum(-1).unique(return_counts=True))

        # dsgv = torch.linalg.svdvals(denoised_adj)
        # for i in dsgv[:100]:
        #     print(f"{i.item():.2e}",end=', ')
        # print('')

        data.adj = denoised_adj.detach().cpu().numpy()
        data.adj = ssp.csr_matrix(data.adj) + ssp.eye(args.num_nodes)

        adj, x, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        import pandas as pd
        result: pd.DataFrame = pd.read_csv(
            f"result/{args.outfile}.csv", index_col=[0, 1])

        gcn = GCN(
            nfeat=args.num_features, nclass=args.num_classes, nhid=16,
            lr=args.lr, weight_decay=args.weight_decay, dropout=0.5, device=device)
        gcn = gcn.to(device)
        gcn.fit(x, adj, labels, idx_train, idx_val, verbose=False,
                patience=30)  # train with earlystopping
        best_test = gcn.test(idx_test)
        logger.info(
            f'GCN on {args.dataset}-{args.ptb_rate:g}, Accuracy: {best_test:.4f}')

        result.loc[(args.dataset, args.ptb_rate),
                   'MFGC-GCN'] = f"{best_test:.4f}"

        pyg_data = Dpr2Pyg(data).process()

        gat = GAT(nfeat=args.num_features,
                  nhid=8, heads=8,
                  nclass=args.num_classes,
                  dropout=0.5, device=device, lr=0.005)
        gat = gat.to(device)
        # train with earlystopping
        gat.fit([pyg_data], patience=100, verbose=False)
        best_test = gat.test()
        logger.info(
            f'GAT on {args.dataset}-{args.ptb_rate:g}, Accuracy: {best_test:.4f}')

        result.loc[(args.dataset, args.ptb_rate),
                   'MFGC-GAT'] = f"{best_test:.4f}"

        # dw = DeepWalk()
        # dw.fit(adj)
        # _,_,_,acc = dw.evaluate_node_classification(labels, idx_train, idx_test)
        # result.loc[(args.dataset, args.ptb_rate),
        #            'MFGC-DW'] = f"{acc:.4f}"

        # nv = Node2Vec()
        # nv.fit(adj)
        # _,_,_,acc = nv.evaluate_node_classification(labels, idx_train, idx_test)
        # result.loc[(args.dataset, args.ptb_rate),
        #            'MFGC-NV'] = f"{acc:.4f}"

        result.to_csv(f"result/{args.outfile}.csv")
