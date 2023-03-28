import argparse


def add_general_group(group):
    group.add_argument("--save_path", type=str, default="results/models/", help="dir path for saving model file")
    group.add_argument("--res_path", type=str, default="results/dict/", help="dir path for output file")
    group.add_argument("--plot_path", type=str, default="results/plot/", help="dir path for output file")
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--mode", type=str, default='clean', help="Mode of running ['clean', 'dp']")
    group.add_argument("--submode", type=str, default='fair', help="")
    group.add_argument("--num_worker", type=int, default=0, help="")
    group.add_argument("--debug", type=bool, default=True)
    group.add_argument("--performance_metric", type=str, default='acc', help="Metrics of performance")


def add_data_group(group):
    group.add_argument('--data_path', type=str, default='Data/', help="dir path to dataset")
    group.add_argument('--dataset', type=str, default='cora', help="name of dataset")
    group.add_argument('--n_neighbor', type=int, default=4, help="# of neighbor each layer")
    group.add_argument('--ratio', type=float, default=0.2, help="train/test split ratio")
    group.add_argument('--folds', type=int, default=5, help='number of folds for cross-validation')


def add_model_group(group):
    group.add_argument("--model_type", type=str, default='sage', help="Model type")
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument('--batch_size', type=int, default=512, help="batch size for training process")
    group.add_argument('--sampling_rate', type=float, default=0.08, help="batch size for training process")
    group.add_argument('--n_hid', type=int, default=2, help='number hidden layer')
    group.add_argument('--hid_dim', type=int, default=32, help='hidden embedding dim')
    group.add_argument("--optimizer", type=str, default='adam')
    group.add_argument("--dropout", type=float, default=0.2)
    group.add_argument("--patience", type=int, default=20)
    group.add_argument("--num_head", type=int, default=8)
    group.add_argument("--aggregator_type", type=str, default='sum')
    group.add_argument("--epochs", type=int, default=100, help='training step')


def add_dp_group(group):
    group.add_argument("--tar_eps", type=float, default=1.0, help="targeted epsilon")
    group.add_argument('--tar_delt', type=float, default=1e-4, help='targeted delta')
    group.add_argument("--ns", type=float, default=1.0, help='noise scale for dp')
    group.add_argument("--clip", type=float, default=1.0, help='clipping gradient bound')
    group.add_argument("--clip_node", type=int, default=4, help='number of allowed appearance')
    group.add_argument("--trim_rule", type=str, default='random', help='trimming rule')


def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    general_group = parser.add_argument_group(title="General configuration")
    dp_group = parser.add_argument_group(title="DP configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    add_dp_group(dp_group)
    return parser.parse_args()
