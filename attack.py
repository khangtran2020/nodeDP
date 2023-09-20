import datetime
import warnings
from config import parse_args_attack
from Data.read import *
from Utils.utils import *
from loguru import logger
from rich import print as rprint
from Attacks.node_attack import run_NMI
from Attacks.link_attack import run_LinkTeller
from Attacks.white_box_attack_grad_norm import run_white_box

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
warnings.filterwarnings("ignore")



def run(args, current_time, device):
    if args.attack_mode == 'node':
        run_NMI(args=args, current_time=current_time, device=device)
    elif args.attack_mode == 'white':
        run_white_box(args=args, current_time=current_time, device=device)
    else:
        run_LinkTeller(args=args, current_time=current_time, device=device)



if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args_attack()
    print_args_attack(args=args)
    args.debug = True if args.debug == 1 else False
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device == 'cpu':
        device = torch.device('cpu')
    rprint(f"DEVICE USING: {device}")
    run(args=args, current_time=current_time, device=device)
