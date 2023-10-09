import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dgl.dataloading import transforms
from dgl.dataloading.base import NID
from Utils.utils import get_index_by_value
from rich import print as rprint


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
    
    def __init__(self, graph, model, num_layer, device, mode='train', weight=None, nnei=-1):
         
        # get nodes
        self.graph = graph.to(device)
        org_nodes = self.graph.nodes()
        mask = 'str_mask' if mode == 'train' else 'ste_mask'
        idx = get_index_by_value(a=self.graph.ndata[mask], val=1)
        self.nodes = org_nodes[idx]
        self.num_layer = num_layer
        self.model = model.to(device)
        self.device = device
        self.membership_label = self.graph.ndata['sha_label'][idx]
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.model.zero_grad()
        self.label_weight = self.membership_label.unique(return_counts=True)
        self.weight = weight
        self.nnei = nnei
        rprint(f"Membership label distribution of shadow {mode}: {self.membership_label.unique(return_counts=True)}")

    def __getitem__(self, index):
        membership_label = self.membership_label[index]
        node = self.nodes[index]
        blocks = self.sample_blocks(seed_nodes=node)
        label = blocks[-1].dstdata["label"]
        out_dict, pred = self.model.forwardout(blocks=blocks, x=blocks[0].srcdata["feat"])
        loss = self.criterion(pred, label)
        loss.backward()
        grad_dict = {}
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                grad_dict[name.replace('.', '-')] = p.grad.clone()
        self.model.zero_grad()
        return (loss, label, out_dict, grad_dict), membership_label

    def __len__(self):

        return self.nodes.size(dim=0)

    def sample_blocks(self, seed_nodes):
        blocks = []
        for i in reversed(range(self.num_layer)):
            if self.weight is not None:
                frontier = self.graph.sample_neighbors(seed_nodes, self.nnei, output_device=self.device, prob=self.weight)
            else:
                frontier = self.graph.sample_neighbors(seed_nodes, self.nnei, output_device=self.device)
            block = transforms.to_block(frontier, seed_nodes, include_dst_in_src=True)
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
        
        return blocks

def custom_collate(batch, out_key, model_key, device, num_class):

    membership_label = torch.Tensor([]).to(device)
    label = torch.Tensor([]).to(device)
    loss = torch.Tensor([]).to(device)
    out_dict = {}
    for key in out_key:
        out_dict[key] = torch.Tensor([]).to(device)
    grad_dict = {}
    for key in model_key:
        grad_dict[key] = torch.Tensor([]).to(device)


    for i, item in enumerate(batch):
        x, y = item
        it_loss, it_label, it_out_dict, it_grad_dict = x

        # membership label
        y = torch.Tensor([(y.item()+1)/2]).to(device)
        membership_label = torch.cat((membership_label, y), dim=0)

        # true label 
        label = torch.cat((label, it_label.detach()), dim=0)

        # loss
        loss = torch.cat((loss, it_loss.detach()), dim=0)

        # out dict
        for key in out_key:
            # rprint(f"At item {i}, out dim of key {key} is {it_out_dict[key].size()}")
            out = torch.unsqueeze(it_out_dict[key], dim=0).detach()
            out_dict[key] = torch.cat((out_dict[key], out), dim=0)

        for key in model_key:
            grad = torch.unsqueeze(it_grad_dict[key], dim=0).detach()
            if 'bias' in key:
                grad = torch.unsqueeze(grad, dim=-1).detach()
            grad_dict[key] = torch.cat((grad_dict[key], grad), dim=0)

    label = F.one_hot(label.long(), num_class).float().to(device)
    loss = torch.unsqueeze(loss, dim=-1)
    for key in model_key:
        grad_dict[key] = torch.unsqueeze(grad_dict[key], dim=1)
    return (label, loss, out_dict, grad_dict), membership_label

    # return filtered_data, filtered_target