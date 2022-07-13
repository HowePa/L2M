import networkx as nx
import dgl
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
from jgrapht.convert import from_nx
from jgrapht.algorithms.matching import blossom5_max_weight


def get_eid_weight(g):
    eid_map = {edge: eid for eid, edge in enumerate(g.edges)}
    nx.set_edge_attributes(g, values=eid_map, name='eid')

    weight_map = {edge: round(np.random.uniform(0, 1), 4) for edge in g.edges}
    nx.set_edge_attributes(g, values=weight_map, name='weight')
    return g


def get_label(g):
    label_map = {edge: 0 for edge in g.edges}
    nx.set_edge_attributes(g, values=label_map, name='label')
    jg = from_nx(g)
    m = blossom5_max_weight(jg)[1]
    for e in m:
        u, v, _ = jg.edge_tuple(e)
        g.edges[u, v]['label'] = 1
    return g


def DLnxtoDGL(G):
    src, dst, eid, weight, label = [], [], [], [], []
    for u, v, arrs in G.edges(data=True):
        src.append(u)
        dst.append(v)
        eid.append(arrs['eid'])
        weight.append(arrs['weight'])
        label.append(arrs['label'])

    src = torch.IntTensor(src)
    dst = torch.IntTensor(dst)
    eid = torch.IntTensor(eid).view(-1, 1)
    weight = torch.DoubleTensor(weight).view(-1, 1)
    label = torch.IntTensor(label).view(-1, 1)

    g = dgl.graph((src, dst))
    g.edata['eid'] = eid
    g.edata['weight'] = weight
    g.edata['label'] = label

    del src, dst, eid, weight, label

    return g


def RLnxtoDGL(G):
    src, dst, eid, weight = [], [], [], []
    for u, v, arrs in G.edges(data=True):
        src.append(u)
        dst.append(v)
        eid.append(arrs['eid'])
        weight.append(arrs['weight'])

    src = torch.IntTensor(src)
    dst = torch.IntTensor(dst)
    eid = torch.IntTensor(eid).view(-1, 1)
    weight = torch.DoubleTensor(weight).view(-1, 1)

    g = dgl.graph((src, dst))
    g.edata['eid'] = eid
    g.edata['weight'] = weight

    del src, dst, eid, weight

    return g


class GraphDataset(Dataset):

    def __init__(self, data_dir=None, generate_fn=None):
        self.data_dir = data_dir
        self.generate_fn = generate_fn
        if data_dir is not None:
            self.data_pathes = [p for p in data_dir.rglob("*.gpickle")]
            self.num_graphs = len(self.data_pathes)
        elif generate_fn is not None:
            self.num_graphs = 200000
        else:
            raise ValueError("Dataset Error!")

    def __getitem__(self, idx):
        if self.generate_fn is None:
            g = nx.read_gpickle(self.data_pathes[idx])
            g = DLnxtoDGL(g)
        else:
            g = self.generate_fn()
            g = RLnxtoDGL(g)

        return g

    def __len__(self):
        return self.num_graphs


def get_dataset(mode='train',
                type='er',
                min_n=50,
                max_n=100,
                er_p=0.15,
                ws_k=4,
                ws_p=0.15,
                ba_m=4,
                hk_m=4,
                hk_p=0.05,
                data_dir=None):
    data_path = Path("Datasets")
    args = {}
    if type == 'er':
        gen = nx.erdos_renyi_graph
        args['p'] = er_p
    elif type == 'ba':
        gen = nx.barabasi_albert_graph
        args['m'] = ba_m
    elif type == 'ws':
        gen = nx.watts_strogatz_graph
        args['k'] = ws_k
        args['p'] = ws_p
    elif type == 'hk':
        gen = nx.powerlaw_cluster_graph
        args['m'] = hk_m
        args['p'] = hk_p
    elif type == 'real':
        # Real Dataset Only
        data_path = data_path / Path(type) / Path(data_dir) / Path(mode)
        if not data_path.exists():
            error_mess = "Not Find Real Dataset at \"{}\"!".format(data_path)
            raise ValueError(error_mess)
        return GraphDataset(data_dir=data_path)
    else:
        error_mess = "Undefined Graph Type <{}>!".format(type)
        raise ValueError(error_mess)

    # Rand Dataset Only
    if mode == 'train':

        def generate_fn():
            num_nodes = np.random.randint(min_n, max_n)
            g = gen(n=num_nodes, **args)
            g = get_eid_weight(g)
            return g

        return GraphDataset(generate_fn=generate_fn)
    else:
        data_dir = type + "_{}_{}".format(min_n, max_n)
        for a in args:
            data_dir = data_dir + "_{}".format(args[a])

        data_path = data_path / Path(type) / Path(data_dir) / Path(mode)
        if not data_path.exists():
            error_mess = "Not Find Rand Dataset at {}!".format(data_path)
            raise ValueError(error_mess)

        return GraphDataset(data_dir=data_path)
