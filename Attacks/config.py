import argparse


def add_general_group(group):
    group.add_argument("--save_path", type=str, default="results/models/", help="dir path for saving model file")
    group.add_argument("--res_path", type=str, default="results/dict/", help="dir path for output file")
    group.add_argument("--plot_path", type=str, default="results/plot/", help="dir path for output file")
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--mode", type=str, default='clean', help="Mode of running ['clean', 'dp']")
    group.add_argument("--submode", type=str, default='none')
    group.add_argument("--device", type=str, default='cpu', help="")
    group.add_argument("--debug", type=int, default=0)
    group.add_argument("--performance_metric", type=str, default='acc', help="Metrics of performance")


def add_data_group(group):
    group.add_argument('--data_path', type=str, default='Data/', help="dir path to dataset")
    group.add_argument('--dataset', type=str, default='cora', help="name of dataset")
    group.add_argument('--n_neighbor', type=int, default=4, help="# of neighbor each layer")


def add_model_group(group):
    group.add_argument("--model_type", type=str, default='sage', help="Model type")
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument('--batch_size', type=int, default=512, help="batch size for training process")
    group.add_argument('--n_layers', type=int, default=2, help='# of layers')
    group.add_argument('--hid_dim', type=int, default=64, help='hidden embedding dim')
    group.add_argument("--optimizer", type=str, default='adam')
    group.add_argument("--dropout", type=float, default=0.2)
    group.add_argument("--patience", type=int, default=20)
    group.add_argument("--num_head", type=int, default=8)
    group.add_argument("--aggregator_type", type=str, default='gcn')
    group.add_argument("--epochs", type=int, default=100, help='training step')
    group.add_argument("--retrain", type=int, default=0)

def add_dp_group(group):
    group.add_argument("--ns", type=float, default=1.0, help='noise scale for dp')
    group.add_argument("--clip", type=float, default=1.0, help='clipping gradient bound')
    group.add_argument("--clip_node", type=int, default=4, help='number of allowed appearance')
    group.add_argument("--trim_rule", type=str, default='adhoc', help='trimming rule')
    group.add_argument('--sampling_rate', type=float, default=0.08, help="batch size for training process")


def add_model_attack_group(group):
    group.add_argument("--att_mode", type=str, default='blackbox', help="Model type")
    group.add_argument("--att_submode", type=str, default='supervised', help="Model type")
    group.add_argument('--att_layers', type=int, default=2, help='# of layers')
    group.add_argument('--att_hid_dim', type=int, default=64, help='hidden embedding dim')
    group.add_argument("--att_lr", type=float, default=0.001, help="learning rate")
    group.add_argument('--att_bs', type=int, default=512, help="batch size for training process")
    group.add_argument("--att_epochs", type=int, default=100, help='training step')
    group.add_argument("--sha_lr", type=float, default=0.001, help="learning rate")
    group.add_argument("--sha_epochs", type=int, default=100, help='training step')
    group.add_argument("--sha_ratio", type=float, default=0.5, help="learning rate")


def parse_args():
    parser = argparse.ArgumentParser()
    exp_grp = parser.add_argument_group(title="Attack setting")

    add_general_group(exp_grp)
    add_data_group(exp_grp)
    add_model_group(exp_grp)
    add_dp_group(exp_grp)
    add_model_attack_group(exp_grp)
    return parser.parse_args()
