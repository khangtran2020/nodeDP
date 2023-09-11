import datetime
import warnings
import sys
from config import parse_args
from Analysis.Struct.run_struct_analysis import run as run_struct
from Analysis.Smooth.run_smooth_analysis import run as run_smooth
from Utils.utils import *
from loguru import logger
from rich import print as rprint

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
warnings.filterwarnings("ignore")

def run(args, current_time, device):

    # init history for running information

    if args.mode == 'clean':
        history = init_history_clean()
    else:
        history = init_history_nodeDP()
    
    if args.analyze_mode == 'struct':
        history['avg_diff_org'] = []
        history['avg_diff_drop'] = []
    elif args.analyze_mode == 'smooth':
        history['va_avg_smooth'] = []
        history['te_avg_smooth'] = []
    save_args_to_history(args=args, history=history)
    name = get_name_analysis(args=args, current_date=current_time)
    history['name'] = name

    run_dict = {
        'struct': run_struct,
        'smooth': run_smooth
    }

    run_mode = run_dict[args.analyze_mode]
    run_mode(args=args, name=name, device=device, history=history)


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

