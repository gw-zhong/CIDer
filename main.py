# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import pickle

from src.utils import *
from torch.utils.data import DataLoader
from src import train
from bert_dataloader import MMDataset

import optuna
from optuna.samplers import TPESampler
from optuna.storages import RetryFailedTrialCallback

def validate_args(args):
    if args.cross_dataset:
        if not args.aligned or args.ood:
            error_msg = []
            if not args.aligned:
                error_msg.append("--aligned must be True")
            if args.ood:
                error_msg.append("--ood must be False")
            raise argparse.ArgumentError(
                argument=None,
                message=f"When --cross_dataset is enabled, the following conditions must be met: {', '.join(error_msg)}"
            )

parser = argparse.ArgumentParser(description='Robust Multimodal Emotion Recognition')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='CIDer',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--ood', action='store_true',
                    help='consider out-of-distribution test or not (default: False)')
parser.add_argument('--cross_dataset', action='store_true',
                    help='consider cross dataset test or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosi',
                    help='dataset to use (default: mosi)')
parser.add_argument('--data_path', type=str, default='/data/zhonggw/MSA_Datasets',
                    help='path for storing the dataset')
parser.add_argument('--model_path', type=str, default='/data/zhonggw/temp_train_models/cider',
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
parser.add_argument('--n_trials', type=int, default=300,
                    help='trials for hyperparameter tuning (default: 300)')
parser.add_argument('--patience', type=int, default=10,
                    help='patience used for early stop (default: 10)')
parser.add_argument('--bias_thresh', type=int, default=100,
                    help='bias word threshold (default: 100)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='do not use cuda (default: False)')
parser.add_argument('--distribute', type=bool, default=False,
                    help='use distributed training (default: False)')
parser.add_argument('--language', type=str, default='en',
                    help='bert language (default: en)')

# Missing case
parser.add_argument('--missing_mode', type=str, default='RMFM',
                    help='modality missing mode (default: RMFM) (only for test)')
parser.add_argument('--missing_rate', type=float, nargs='+', default=[0.0],
                    help='missing rates (only for test)')

args = parser.parse_args()

try:
    validate_args(args)
except argparse.ArgumentError as e:
    parser.error(str(e))

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

# If you are running low on memory, it is recommended to set pin_memory=False
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

print(f'### Note: You are running in BERT mode.')

if __name__ == '__main__':
    sampler = TPESampler(seed=args.seed)
    study_name = f'{cross_dataset_status}[{hyp_params.dataset}]task-{hyp_params.task}-trials-{hyp_params.n_trials}'
    url = f'sqlite:///{cross_dataset_status}{args.dataset}-{aligned_status}-{ood_status}.db'
    print('Start Hyperparameter Tuning....')
    print(f'Total Trials Are {hyp_params.n_trials}....')
    print(f'Your Database Url is {url}....')

    def objective(trial):
        hyp_params.multimodal_layers = trial.suggest_int('multimodal_layers', 1, 4, step=1)
        hyp_params.num_heads = trial.suggest_categorical('num_heads', choices=[1, 2, 4, 8])
        hyp_params.attn_dropout = trial.suggest_float('attn_dropout', 0.0, 0.5, step=0.1)
        hyp_params.out_dropout = trial.suggest_float('out_dropout', 0.0, 0.5, step=0.1)
        hyp_params.embed_dropout = trial.suggest_float('embed_dropout', 0.0, 0.5, step=0.1)
        hyp_params.joint_rep_weight = trial.suggest_float('joint_rep_weight', 0.0, 1.0, step=0.1)
        hyp_params.attn_weight = trial.suggest_float('attn_weight', 0.0, 1.0, step=0.1)
        hyp_params.recon_weight = trial.suggest_float('recon_weight', 0.0, 1.0, step=0.1)

        ans = train.initiate(hyp_params, train_loader, valid_loader, test_loader, trial, device)

        return ans

    last_best_trial_number = 0
    last_best_trial_value = 0.0
    def delete_models(study, trial):  # delete temporary tuning models
        global last_best_trial_number
        global last_best_trial_value
        if trial.number == 0:
            last_best_trial_value = study.best_trial.value
        else:
            if study.best_trial.value <= last_best_trial_value:
                # delete current model
                model_name = save_load_name(hyp_params, names={'model_name': hyp_params.model,
                                                               'trial_number': trial.number})
                model_name = f'{hyp_params.model_path}/{model_name}.pt'
                if os.path.exists(model_name):
                    os.remove(model_name)
            else:  # study.best_trial.value > last_best_trial_value (current model is better)
                # delete previous best model
                model_name = save_load_name(hyp_params, names={'model_name': hyp_params.model,
                                                               'trial_number': last_best_trial_number})
                model_name = f'{hyp_params.model_path}/{model_name}.pt'
                if os.path.exists(model_name):
                    os.remove(model_name)
                last_best_trial_number = trial.number
                last_best_trial_value = study.best_trial.value

    study = optuna.create_study(
        direction='maximize', sampler=sampler,
        study_name=study_name, storage=url, load_if_exists=True)
    study.optimize(objective, n_trials=hyp_params.n_trials, callbacks=[delete_models])
    best_trial = study.best_trial

    old_name = save_load_name(hyp_params, names={'model_name': hyp_params.model,
                                                 'trial_number': best_trial.number})
    old_name = f'{hyp_params.model_path}/{old_name}.pt'
    new_name = old_name.split('_trial_')[0] + '.pt'
    os.rename(old_name, new_name)
    print(f'Rename {old_name} to {new_name}...')

    print('-' * 50)
    print(f'Best Accuracy is: {best_trial.value:.4f}')
    print(f'Best Hyperparameters are:\n{best_trial.params}')

    df = study.trials_dataframe().to_csv(
        'results/{}{}_{}_{}_{}_trialNum_{}.csv'.format(
            cross_dataset_status, dataset, aligned_status, ood_status, hyp_params.task, hyp_params.n_trials))