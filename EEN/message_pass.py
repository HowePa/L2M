import dgl
import dgl.function as fn


def update_efeat(G, feat):
    rG = dgl.reverse(G)
    G.edata['h'] = feat.double()
    rG.edata['h'] = feat.double()
    # send message to nodes
    G.update_all(fn.copy_e('h', 'm'), fn.sum('m', 'h'))
    rG.update_all(fn.copy_e('h', 'm'), fn.sum('m', 'h'))
    G.ndata['h'] += rG.ndata.pop('h')
    # aggregate message from nodes
    G.apply_edges(lambda edges: {'h': edges.src['h'] + edges.dst['h'] - 2 * edges.data['h']})
    G.ndata.pop('h')
    del rG
    return G.edata.pop('h')


def sum_efeat(G, feat):
    G.edata['sum'] = feat.double()
    ret = dgl.readout_edges(G, 'sum')
    G.edata.pop('sum')
    return ret


def max_efeat(G, feat):
    G.edata['max'] = feat.double()
    ret = dgl.readout_edges(G, 'max', op='max')
    G.edata.pop('max')
    return ret
