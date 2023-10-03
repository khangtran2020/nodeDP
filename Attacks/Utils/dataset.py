import torch
from torch.utils.data import Dataset
from dgl.dataloading import transforms
from dgl.dataloading.base import NID
from Utils.utils import get_index_by_value


class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = self.X.size(dim=0)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

class ShadowData(Dataset):
    
    def __init__(self, graph, model, num_layer, device, mode='train'):
         
        # get nodes
        self.graph = graph.to(device)
        org_nodes = self.graph.nodes()
        mask = 'str_mask' if mode == 'train' else 'ste_mask'
        idx = get_index_by_value(a=mask, val=1)
        self.nodes = org_nodes[idx]
        self.num_layer = num_layer
        self.model = model.to(device)
        self.device = device
        self.membership_label = self.graph.ndata['sha_label'][idx]
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.model.zero_grad()

    def __getitem__(self, index):

        membership_label = self.membership_label[index]
        node = self.nodes[index]
        blocks = self.sample_blocks(seed_nodes=node)
        label = blocks[-1].dstdata["label"]
        out_dict, pred = self.model.forwardout(blocks=blocks, x=blocks.srcdata["feat"])
        loss = self.criterion(pred, label)
        loss.backward()
        grad_dict = {}
        for name, p in self.model.named_parameters():
            grad_dict[name] = p.grad
        self.model.zero_grad()
        return (loss, label, out_dict, grad_dict), membership_label

    def __len__(self):

        return self.nodes.size(dim=0)

    def sample_blocks(self, seed_nodes):

        blocks = []

        for i in reversed(range(self.num_layer)):
            frontier = self.graph.sample_neighbors(seed_nodes, -1, output_device=self.device)
            block = transforms.to_block(frontier, seed_nodes, include_dst_in_src=True)
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
        
        return blocks




def custom_collate(batch, out_key, model_key, device):
    pass
    # membership_label = torch.Tensor([]).to(device)
    # label = torch.Tensor([]).to(device)
    # loss = torch.Tensor([]).to(device)
    # out_dict = {}
    # for key in out_key:
    #     out_dict[key] = torch.Tensor([]).to(device)
    # grad_dict = {}
    # for key in model_key:
    #     grad_dict[key] = torch.Tensor([]).to(device)


    # for item in batch:
    #     x, y = item
    #     it_loss, it_label, it_out_dict, it_grad_dict = x
    

    # return filtered_data, filtered_target