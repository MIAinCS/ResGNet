from argparse import ArgumentParser


parser = ArgumentParser()

# experiment setup
parser.add_argument("--exp", type=str, default="default", required=False)
parser.add_argument("--visdom", type=int, default=0, required=False)
parser.add_argument("--dim", type=int, default=3, required=False)
parser.add_argument("--fold", type=int, default=5, required=False)
parser.add_argument("--num_exp", type=int, default=5, required=False)
parser.add_argument("--data", type=str, default="")
parser.add_argument("--grad", type=int, default=0)
parser.add_argument("--isbi", type=int, default=0, required=False)
parser.add_argument("--category", type=str, choices=["CG", "PZ"], required=False)

# train params
parser.add_argument("--lr", type=float, default=1e-3, required=False)
parser.add_argument("--weight_decay", type=float, default=1e-6, required=False)
parser.add_argument("--warm_up", type=int, default=200, required=False)
parser.add_argument("--net", type=str, default="unet", required=False)
parser.add_argument("--epoch", type=int, default=100, required=False)
parser.add_argument("--train", type=int, default=1, required=False)
parser.add_argument("--optim", type=str, default="adam", required=False)
parser.add_argument("--batch_size", type=int, default=1, required=False)

# network params
parser.add_argument("--activation", type=str, default="leaky_relu", required=False)
parser.add_argument("--dropout", type=float, default=0, required=False)

config = vars(parser.parse_args())