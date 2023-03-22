import dgl

def read_data(data_name, ratio):
    if data_name == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif data_name == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif data_name == 'pubmed':
        data = dgl.data.PubmedGraphDataset()

    g_train, g_val, g_test = train_test_split(data=data, ratio=ratio)
    return g_train, g_val, g_test

def train_test_split(data, ratio):
    g = data[0]
    train_idx, val_idx, test_idx = dgl.data.utils.split_dataset(
        g, frac_list=ratio, shuffle=True)
    train_g = g.subgraph(train_idx.indices)
    val_g = g.subgraph(val_idx.indices)
    test_g = g.subgraph(test_idx.indices)
    return train_g, val_g, test_g