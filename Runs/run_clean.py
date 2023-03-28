from dgl.dataloading import NeighborSampler

def run(dataloaders, model, optimizer, name):
    tr_loader, val_loader, te_loader = dataloaders

def update_onestep(model, optimizer, batch):
    input_nodes, output_nodes, mfgs = batch
    