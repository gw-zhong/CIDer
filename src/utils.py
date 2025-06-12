import torch
import os
import random
import numpy as np


def save_load_name(args, names):
    aligned_status = 'unaligned'
    ood_status = 'iid'
    cross_dataset_status = ''
    model_name = names['model_name']
    trial_number = names['trial_number']
    if args.aligned:
        aligned_status = 'aligned'
    if args.ood:
        ood_status = 'ood'
    if args.cross_dataset:
        cross_dataset_status = '[cross_dataset]'
    return f'{cross_dataset_status}{args.dataset}_{aligned_status}_{ood_status}_{args.task}_{model_name}_trial_{trial_number}'


def save_model(args, model, names):
    name = save_load_name(args, names)
    torch.save(model, f'{args.model_path}/{name}.pt')
    print(f"Saved model at {args.model_path}/{name}.pt!")


def load_model(args, names):
    name = save_load_name(args, names)
    print(f"Loading model at {args.model_path}/{name}.pt!")
    model = torch.load(f'{args.model_path}/{name}.pt')
    return model


def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
