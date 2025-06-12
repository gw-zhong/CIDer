import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train_eval
from bert_dataloader import MMDataset

parser = argparse.ArgumentParser(description='Robust Multimodal Emotion Recognition')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='CIDer',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--aligned', type=bool, default=False,
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--ood', type=bool, default=True,
                    help='consider out-of-distribution test or not (default: True)')
parser.add_argument('--cross_dataset', type=bool, default=False,
                    help='consider cross dataset test or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei',
                    help='dataset to use (default: mosi)')
parser.add_argument('--data_path', type=str, default='/data/zhonggw/MSA_Datasets',
                    help='path for storing the dataset')
parser.add_argument('--model_path', type=str, default='/data/zhonggw/temp_train_models/cider',
                    help='path for storing the model')
parser.add_argument('--task', type=str, default='seven',
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

args.language = 'en'
args.weight_decay_bert = 0.001
args.lr_bert = 5e-5
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
    test_set = 'mosei' if args.dataset == 'mosi' else 'mosi'
    print(f"### Note: You are running in cross-dataset (test set - {test_set}) mode.")
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
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 5, 20
        if args.aligned:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 50, 50
        else:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 375, 500
    else:
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 74, 35
        if args.aligned:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 50, 50
        else:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 500, 500
else:
    raise ValueError('Unknown dataset')

hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(f'{dataset}_{hyp_params.task}', 1)
hyp_params.criterion = criterion_dict.get(f'{dataset}_{hyp_params.task}', 'L1Loss')
hyp_params.recon_loss = 'SmoothL1Loss'
hyp_params.embed_dim = 32

print(f'### Note: You are running in BERT mode.')
missing_rate_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# missing_rate_list = [0.]
#############################
# Specific Modality Missing #
#############################
# missing_rate_list = [[0., 1.0, 1.0],  # (Text)
#                      [1.0, 0., 1.0],  # (Audio)
#                      [1.0, 1.0, 0.],  # (Vision)
#                      [0., 0., 1.0],  # (Text, Audio)
#                      [0., 1.0, 0.],  # (Text, Vision)
#                      [1.0, 0., 0.],  # (Audio, Vision)
#                      [0., 0., 0.], ]  # (Text, Audio, Vision)
Non0_acc_2_list = []
Non0_F1_score_list = []
Acc_7_list = []
if __name__ == '__main__':
    if not os.path.exists(
            f'./{cross_dataset_status}{args.dataset}_bias_{args.bias_thresh}_{ood_status}_{args.task}_words_dict.pkl'):  # Just use training set to create the bias word dict
        train_data = MMDataset(args, 'train')
    print(f'### Missing Mode - {args.missing_mode}')
    print('-' * 50)
    for mr in missing_rate_list:
        print(f'### Missing Rate - {mr}')
        args.missing_rate = [mr, mr, mr] if args.missing_mode == 'RMFM_same' else [mr]
        # args.missing_rate = mr  # Just for Specific Modality Missing
        print("Start loading the test data....")
        test_data = MMDataset(args, 'test')
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=14, persistent_workers=True, pin_memory=True)
        print('Finish loading the test data....')

        if hyp_params.cross_dataset:
            print('Start CROSS-DATASET Testing....')
        else:
            print('Start Testing....')

        ans = train_eval.initiate(hyp_params, test_loader, device)
        if task == 'binary':
            Non0_acc_2_list.append(ans['Non0_acc_2'])
            Non0_F1_score_list.append(ans['Non0_F1_score'])
        else:
            Acc_7_list.append(ans['Acc_7'])

    ###########################################################################################
    # if you want to test Specific Modality Missing, you must comment out the following code. #
    ###########################################################################################
    print('Calculate AUILC...')
    if task == 'binary':
        non0_acc_2_auilc = 0
        for i in range(10):
            non0_acc_2_auilc += (Non0_acc_2_list[i] + Non0_acc_2_list[i + 1]) * (missing_rate_list[i + 1] - missing_rate_list[i]) / 2
        non0_f1_score_auilc = 0
        for i in range(10):
            non0_f1_score_auilc += (Non0_F1_score_list[i] + Non0_F1_score_list[i + 1]) * (missing_rate_list[i + 1] - missing_rate_list[i]) / 2
        print(f'Non0_acc_2_AUILC: {non0_acc_2_auilc:.3f}')
        print(f'Non0_F1_score_AUILC: {non0_f1_score_auilc:.3f}')
    else:
        acc_7_auilc = 0
        for i in range(10):
            acc_7_auilc += (Acc_7_list[i] + Acc_7_list[i + 1]) * (missing_rate_list[i + 1] - missing_rate_list[i]) / 2
        print(f'Acc_7_AUILC: {acc_7_auilc:.3f}')
