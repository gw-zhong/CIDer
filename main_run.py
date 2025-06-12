import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train_run
from bert_dataloader import MMDataset

import optuna
from optuna.samplers import TPESampler

parser = argparse.ArgumentParser(description='Robust Multimodal Emotion Recognition')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='CIDer',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--aligned', type=bool, default=False,
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--ood', type=bool, default=False,
                    help='consider out-of-distribution test or not (default: True)')
parser.add_argument('--cross_dataset', type=bool, default=False,
                    help='consider cross dataset test or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosi',
                    help='dataset to use (default: mosi)')
parser.add_argument('--data_path', type=str, default='/data/zhonggw/MSA_Datasets',
                    help='path for storing the dataset')
parser.add_argument('--model_path', type=str, default='/data/zhonggw/temp_train_models',
                    help='path for storing the model')
parser.add_argument('--task', type=str, default='binary',
                    help='binary/seven classification task')

# Dropouts
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')

# Tuning
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs (default: 200)')

# Logistics
parser.add_argument('--patience', type=int, default=10,
                    help='patience used for early stop (default: 10)')
parser.add_argument('--bias_thresh', type=int, default=100,
                    help='bias word threshold (default: 100)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='do not use cuda (default: false)')
parser.add_argument('--distribute', type=bool, default=False,
                    help='use distributed training (default: false)')
parser.add_argument('--language', type=str, default='en',
                    help='bert language (default: en)')

# Missing case
parser.add_argument('--missing_mode', type=str, default='RMFM',
                    help='modality missing mode (default: RMFM) (only for test)')
parser.add_argument('--missing_rate', type=float, nargs='+', default=[0.0],
                    help='missing rates (only for test)')

args = parser.parse_args()

seed_everything(args)
dataset = str.lower(args.dataset.strip())
task = args.task

use_cuda = False

output_dim_dict = {
    'mosi_binary': 3,
    'mosei_binary': 3,
    'mosi_seven': 7,
    'mosei_seven': 7
}

criterion_dict = {
    'mosi_binary': 'NLLLoss',  # classification
    'mosei_binary': 'NLLLoss',
    'mosi_seven': 'NLLLoss',  # classification
    'mosei_seven': 'NLLLoss'
}

torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        device = torch.device('cpu')
    else:
        seed_everything(args)
        use_cuda = True
        device = torch.device('cuda')
else:
    device = torch.device('cpu')

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

args.language = 'en'
args.weight_decay_bert = 0.001
args.lr_bert = 5e-5
train_data = MMDataset(args, 'train')
valid_data = MMDataset(args, 'valid')
test_data = MMDataset(args, 'test')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=14, persistent_workers=True, pin_memory=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=14, persistent_workers=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=14, persistent_workers=True, pin_memory=True)

print('Finish loading the data....')
print(f'### Dataset - {dataset}')
print(f'### Classification Task - {args.task}')
aligned_status = 'aligned'
if not args.aligned:
    print("### Note: You are running in unaligned mode.")
    aligned_status = 'unaligned'
else:
    print("### Note: You are running in aligned mode.")
ood_status = 'iid'
if args.ood:
    print("### Note: You are running in OOD mode.")
    ood_status = 'ood'
else:
    print("### Note: You are running in IID mode.")
cross_dataset_status = ''
if args.cross_dataset:
    print("### Note: You are running in cross-dataset mode.")
    cross_dataset_status = '[cross_dataset]'

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args

# Fix feature dimensions and sequence length
if dataset == 'mosi' or dataset == 'mosei':
    if dataset == 'mosi':
        if not hyp_params.cross_dataset:
            hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 5, 20
            if args.aligned:
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 50, 50
            else:
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 375, 500
        else:
            hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 25, 171
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 44, 44, 44
    else:
        if not hyp_params.cross_dataset:
            hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 74, 35
            if args.aligned:
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 50, 50
            else:
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 500, 500
        else:
            hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 25, 171
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 75, 75, 75
else:
    raise ValueError('Unknown dataset')

hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(f'{dataset}_{hyp_params.task}', 1)
hyp_params.criterion = criterion_dict.get(f'{dataset}_{hyp_params.task}', 'L1Loss')
hyp_params.recon_loss = 'SmoothL1Loss'
hyp_params.embed_dim = 32
# Fixed hyperparameters
hyp_params.multimodal_layers = 3
hyp_params.num_heads = 8
hyp_params.attn_dropout = 0.0
hyp_params.out_dropout = 0.30000000000000004
hyp_params.embed_dropout = 0.2
hyp_params.joint_rep_weight = 0.5
hyp_params.attn_weight = 0.1
hyp_params.recon_weight = 0.6000000000000001

print(f'### Note: You are running in BERT mode.')

if __name__ == '__main__':
    print('Start Training (Once)....')
    ans = train_run.initiate(hyp_params, train_loader, valid_loader, test_loader, device)