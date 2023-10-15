import sys
import torch
from loguru import logger
from rich import print as rprint
from Utils.utils import timeit
from Attacks.Utils.utils import save_dict
from Attacks.Utils.data_utils import generate_nohop_graph
from Attacks.Utils.train_eval import retrain
from Utils.console import console
from Utils.utils import get_index_by_value
from functools import partial


logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def run(args, graph, model, device, history, name):

    train_g, val_g, test_g, shadow_graph = graph
    model_hist, att_hist = history

    with timeit(logger=logger, task='init-target-model'):

        if args.exist_model == False:
            rprint(f"Model is {args.exist_model} to exist, need to retrain")
            model_name = f"{name['model']}.pt"
            model, model_hist = retrain(args=args, train_g=train_g, val_g=val_g, test_g=test_g, model=model, 
                                        device=device, history=model_hist, name=model_name[:-3])
            target_model_name = f"{name['model']}.pkl"
            target_model_path = args.res_path + target_model_name
            save_dict(path=target_model_path, dct=model_hist)
        
    with timeit(logger=logger, task='preparing-link-data'):
        
        shadow_graph = shadow_graph.to(device)
        shadow_graph_nohop = generate_nohop_graph(graph=shadow_graph, device=device)

        src_edge, dst_edge = shadow_graph.edges()

        # get edges in shadow train & test
        str_mask = shadow_graph.ndata['str_mask']
        ste_mask = shadow_graph.ndata['ste_mask']

        src_edge_intr = str_mask[src_edge]
        dst_edge_intr = str_mask[dst_edge]
        mask_intr = torch.logical_and(src_edge_intr, dst_edge_intr).int()
        idx_edge_intr = get_index_by_value(a=mask_intr, val=1)

        src_edge_inte = ste_mask[src_edge]
        dst_edge_inte = ste_mask[dst_edge]
        mask_inte = torch.logical_and(src_edge_inte, dst_edge_inte).int()
        idx_edge_inte = get_index_by_value(a=mask_inte, val=1)

        console.log(f"Index in train & test: {torch.isin(idx_edge_intr, idx_edge_inte).sum().item() + torch.isin(idx_edge_inte, idx_edge_intr).sum().item()}")

        # get edges in the same set in train 
        pos_mask_tr = shadow_graph.ndata['pos_mask_tr']
        neg_mask_tr = shadow_graph.ndata['neg_mask_tr']

        src_edge_pos_intr = pos_mask_tr[src_edge]
        dst_edge_pos_intr = pos_mask_tr[dst_edge]
        mask_pos_intr = torch.logical_and(src_edge_pos_intr, dst_edge_pos_intr).int()
        indx_pos_intr = get_index_by_value(a=mask_pos_intr, val=1)

        src_edge_neg_intr = neg_mask_tr[src_edge]
        dst_edge_neg_intr = neg_mask_tr[dst_edge]
        mask_neg_intr = torch.logical_and(src_edge_neg_intr, dst_edge_neg_intr).int()
        indx_neg_intr = get_index_by_value(a=mask_neg_intr, val=1)

        indx_same_intr = torch.cat((indx_pos_intr, indx_neg_intr), dim=0)

        console.log(f"Index in train pos & train neg: {torch.isin(indx_pos_intr, indx_neg_intr).sum().item() + torch.isin(indx_neg_intr, indx_pos_intr).sum().item()}")
        console.log(f"Index in train same in train: {torch.isin(indx_same_intr, idx_edge_intr).sum().item() - indx_same_intr.size(dim=0)}")

        # get edges in diff set in train
        mask_pos_neg_intr = torch.logical_and(src_edge_pos_intr, dst_edge_neg_intr).int()
        indx_pos_neg_intr = get_index_by_value(a=mask_pos_neg_intr, val=1)

        mask_neg_pos_intr = torch.logical_and(src_edge_neg_intr, dst_edge_pos_intr).int()
        indx_neg_pos_intr = get_index_by_value(a=mask_neg_pos_intr, val=1)

        indx_diff_intr = torch.cat((indx_pos_neg_intr, indx_neg_pos_intr), dim=0)

        console.log(f"Index in train pos neg & train neg pos: {torch.isin(indx_pos_neg_intr, indx_neg_pos_intr).sum().item() + torch.isin(indx_neg_pos_intr, indx_pos_neg_intr).sum().item()}")
        console.log(f"Index in train diff in train: {torch.isin(indx_diff_intr, idx_edge_intr).sum().item() - indx_diff_intr.size(dim=0)}")
        
        # get edges in the same set in test 
        pos_mask_te = shadow_graph.ndata['pos_mask_te']
        neg_mask_te = shadow_graph.ndata['neg_mask_te']

        src_edge_pos_inte = pos_mask_te[src_edge]
        dst_edge_pos_inte = pos_mask_te[dst_edge]
        mask_pos_inte = torch.logical_and(src_edge_pos_inte, dst_edge_pos_inte).int()
        indx_pos_inte = get_index_by_value(a=mask_pos_inte, val=1)

        src_edge_neg_inte = neg_mask_te[src_edge]
        dst_edge_neg_inte = neg_mask_te[dst_edge]
        mask_neg_inte = torch.logical_and(src_edge_neg_inte, dst_edge_neg_inte).int()
        indx_neg_inte = get_index_by_value(a=mask_neg_inte, val=1)

        indx_same_inte = torch.cat((indx_pos_inte, indx_neg_inte), dim=0)

        console.log(f"Index in test pos & test neg: {torch.isin(indx_pos_inte, indx_neg_inte).sum().item() + torch.isin(indx_neg_inte, indx_pos_inte).sum().item()}")
        console.log(f"Index in test same in test: {torch.isin(indx_same_inte, idx_edge_inte).sum().item() - indx_same_inte.size(dim=0)}")

        # get edges in diff set in test
        mask_pos_neg_inte = torch.logical_and(src_edge_pos_inte, dst_edge_neg_inte).int()
        indx_pos_neg_inte = get_index_by_value(a=mask_pos_neg_inte, val=1)

        mask_neg_pos_inte = torch.logical_and(src_edge_neg_inte, dst_edge_pos_inte).int()
        indx_neg_pos_inte = get_index_by_value(a=mask_neg_pos_inte, val=1)

        indx_diff_inte = torch.cat((indx_pos_neg_inte, indx_neg_pos_inte), dim=0)

        console.log(f"Index in test pos neg & test neg pos: {torch.isin(indx_pos_neg_inte, indx_neg_pos_inte).sum().item() + torch.isin(indx_neg_pos_inte, indx_pos_neg_inte).sum().item()}")
        console.log(f"Index in test diff in test: {torch.isin(indx_diff_inte, idx_edge_inte).sum().item() - indx_diff_inte.size(dim=0)}")
        
    sys.exit()   

    return model_hist, att_hist

