import datetime
import warnings
import sys
from config import parse_args
from Data.read import read_data_link
from Models.init import init_optimizer
from Utils.utils import *
from loguru import logger
from rich import print as rprint
from Models.models import GraphSageGraph, DotPredictor
from Models.train_eval import EarlyStopping
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
warnings.filterwarnings("ignore")

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def run(args, current_time, device):


    history = init_history_link()
    model_name = f'{args.dataset}_link_predition.pt'

    with timeit(logger, 'init-data'):
        train_g, tr_data, te_data = read_data_link(args=args, data_name=args.dataset, history=history)
        train_pos_g, train_neg_g = tr_data
        test_pos_g, test_neg_g = te_data

    model = GraphSageGraph(train_g.ndata['feat'].shape[1], 16)
    pred = DotPredictor()
    optimizer = init_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)
    es = EarlyStopping(patience=args.patience, verbose=False, mode='min')
    all_logits = []
    for e in range(100):
        # forward
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        es(epoch=e, epoch_score=loss.item(), model=model, model_path=args.save_path + model_name)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {}'.format(e, loss))

    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print('AUC', compute_auc(pos_score, neg_score))


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    print_args(args=args)
    args.debug = True if args.debug == 1 else False
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device == 'cpu':
        device = torch.device('cpu')
    rprint(f"DEVICE USING: {device}")
    run(args=args, current_time=current_time, device=device)
