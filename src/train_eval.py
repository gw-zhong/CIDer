import numpy as np
from torch import nn
from src.utils import *
from src.eval_metrics import *

eval_metric = 'Non0_acc_2'


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, test_loader, device):
    if hyp_params.task == 'seven':
        global eval_metric
        eval_metric = 'Acc_7'
    settings = {'task': hyp_params.task}
    return train_model(settings, hyp_params, test_loader, device)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, test_loader, device):
    task = settings['task']

    def evaluate(model):
        model.eval()
        loader = test_loader

        results = []
        cf_results = []
        truths = []
        joint_reps_m = []
        video_ids = []

        with torch.no_grad():
            for batch_data in loader:
                text = batch_data['text'].to(device)
                audio = batch_data['audio'].to(device)
                vision = batch_data['vision'].to(device)
                text_m = batch_data['text_m'].to(device)
                audio_m = batch_data['audio_m'].to(device)
                vision_m = batch_data['vision_m'].to(device)
                binary_labels = batch_data['labels']['binary'].to(device)
                seven_labels = batch_data['labels']['seven'].to(device)
                text_mask = batch_data['text_mask'].to(device)
                audio_mask = batch_data['audio_mask'].to(device)
                vision_mask = batch_data['vision_mask'].to(device)
                text_lengths = batch_data['text_lengths'].to(device)
                cls_probs = batch_data['cls_probs'].to(device)
                text_cf = batch_data['text_cf'].to(device)
                binary_labels = binary_labels.view(-1).long()
                seven_labels = seven_labels.view(-1).long()
                cls_ids = 3 if task == 'binary' else 7
                cls_feats = []

                for idx in range(cls_ids):
                    cls_feats.append(batch_data['cls_feats'][idx].to(device))

                net = nn.DataParallel(model) if hyp_params.distribute else model
                outputs_m = net(**dict(text=(text, text_m, text_cf), audio=(audio, audio_m), vision=(vision, vision_m),
                                       input_masks=(text_mask, audio_mask, vision_mask),
                                       text_lengths=text_lengths, cls_feats=cls_feats, cls_probs=cls_probs,
                                       device=device, test=True, missing=True))

                # Collect the results into dictionary
                results.append(outputs_m['pred_m'])
                cf_results.append(outputs_m['cf_pred_m'])
                joint_reps_m.append(outputs_m['joint_rep_m'])
                video_ids.extend(batch_data['id'])
                if task == 'binary':
                    truths.append(binary_labels)
                else:  # seven
                    truths.append(seven_labels)

        results = torch.cat(results)
        cf_results = torch.cat(cf_results)
        joint_reps_m = torch.cat(joint_reps_m)
        truths = torch.cat(truths)

        return results, cf_results, truths, joint_reps_m, video_ids

    best_model_name = save_load_name(hyp_params, names={'model_name': hyp_params.model,
                                                        'trial_number': 140})
    best_model_name = best_model_name.split('_trial_')[0]
    print(f"Loading model at {hyp_params.model_path}/{best_model_name}.pt!")
    best_model = torch.load(f'{hyp_params.model_path}/{best_model_name}.pt')
    total_parameters = sum([param.nelement() for param in best_model.parameters()])
    bert_parameters = sum([param.nelement() for param in best_model.text_model.parameters()])
    print(f'Trainable Parameters: {total_parameters}...')
    print(f'BERT Parameters: {bert_parameters}...')
    print(f'CIDer Parameters: {total_parameters - bert_parameters}...')
    results, cf_results, truths, joint_reps_m, video_ids = evaluate(best_model)

    if hyp_params.dataset == 'mosei':
        if hyp_params.task == 'binary':
            ans = eval_mosei_classification_binary_cf(results, cf_results, truths)
        else:  # seven
            ans = eval_mosei_classification_seven_cf(results, cf_results, truths)
    else:  # mosi
        if hyp_params.task == 'binary':
            ans = eval_mosi_classification_binary_cf(results, cf_results, truths)
        else:  # seven
            ans = eval_mosi_classification_seven_cf(results, cf_results, truths)

    return ans


class ReconLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.eps = 1e-6
        self.type = type
        if type == 'L1Loss':
            self.loss = nn.L1Loss(reduction='sum')
        elif type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2]).float()

        loss = self.loss(pred * mask, target * mask) / (torch.sum(mask) + self.eps)

        return loss
