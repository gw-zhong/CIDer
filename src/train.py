import optuna
import torch
import torch.nn.functional as F
from torch import nn
import sys
import csv
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import math
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *

eval_metric = 'Non0_acc_2'


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader, trial, device):
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)
    if hyp_params.task == 'seven':
        global eval_metric
        eval_metric = 'Acc_7'

    if hyp_params.use_cuda:
        model = model.to(device)

    bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_params = list(model.text_model.named_parameters())
    bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
    bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
    model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]
    optimizer_grouped_parameters = [
        {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
        {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
        {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters)
    task_criterion = getattr(nn, hyp_params.criterion)()
    recon_criterion = ReconLoss(hyp_params.recon_loss)
    patience = hyp_params.patience
    task = hyp_params.task
    settings = {'model': model,
                'optimizer': optimizer,
                'task': task,
                'task_criterion': task_criterion,
                'recon_criterion': recon_criterion,
                'patience': patience}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader, trial, device)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader, trial, device):
    model = settings['model']
    optimizer = settings['optimizer']
    task = settings['task']
    task_criterion = settings['task_criterion']
    recon_criterion = settings['recon_criterion']
    patience = settings['patience']

    def train(model, optimizer, task_criterion, recon_criterion):
        epoch_loss = 0
        model.train()

        for batch_data in train_loader:
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
            text_missing_mask = batch_data['text_missing_mask'].to(device)
            audio_missing_mask = batch_data['audio_missing_mask'].to(device)
            vision_missing_mask = batch_data['vision_missing_mask'].to(device)
            text_lengths = batch_data['text_lengths'].to(device)
            cls_probs = batch_data['cls_probs'].to(device)
            binary_labels = binary_labels.view(-1).long()
            seven_labels = seven_labels.view(-1).long()
            cls_ids = 3 if task == 'binary' else 7
            cls_feats = []

            batch_size = text.size(0)

            for idx in range(cls_ids):
                cls_feats.append(batch_data['cls_feats'][idx].to(device))

            model.zero_grad()

            net = nn.DataParallel(model) if hyp_params.distribute else model
            outputs = net(**dict(text=(text, text_m, None), audio=(audio, audio_m), vision=(vision, vision_m),
                                 input_masks=(text_mask, audio_mask, vision_mask),
                                 text_lengths=text_lengths, cls_feats=cls_feats, cls_probs=cls_probs,
                                 device=device, test=False, missing=False))

            if task == 'binary':
                loss_task = task_criterion(outputs['pred'], binary_labels)
            else:  # seven
                loss_task = task_criterion(outputs['pred'], seven_labels)
            loss_task.backward()
            optimizer.step()

            outputs_m = net(**dict(text=(text, text_m, None), audio=(audio, audio_m), vision=(vision, vision_m),
                                   input_masks=(text_mask, audio_mask, vision_mask),
                                   text_lengths=text_lengths, cls_feats=cls_feats, cls_probs=cls_probs,
                                   device=device, test=False, missing=True))

            if task == 'binary':
                loss_task_m = task_criterion(outputs_m['pred_m'], binary_labels)
            else:  # seven
                loss_task_m = task_criterion(outputs_m['pred_m'], seven_labels)

            loss_joint_rep = 1 - F.cosine_similarity(outputs_m['joint_rep_m'], outputs['joint_rep'].detach(),
                                                     dim=-1).mean()

            loss_attn = F.kl_div(F.log_softmax(outputs_m['attn_m'], dim=-1),
                                 F.softmax(outputs['attn'].detach(), dim=-1), reduction='batchmean')

            mask = text_mask - text_missing_mask
            loss_recon_text = recon_criterion(outputs_m['text_recon'], outputs['text_for_recon'], mask)
            mask = audio_mask - audio_missing_mask
            loss_recon_audio = recon_criterion(outputs_m['audio_recon'], audio, mask)
            mask = vision_mask - vision_missing_mask
            loss_recon_video = recon_criterion(outputs_m['vision_recon'], vision, mask)
            loss_recon = loss_recon_text + loss_recon_audio + loss_recon_video

            combined_loss = loss_task_m + \
                            hyp_params.joint_rep_weight * loss_joint_rep + \
                            hyp_params.attn_weight * loss_attn + \
                            hyp_params.recon_weight * loss_recon
            combined_loss.backward()
            optimizer.step()

            epoch_loss += combined_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, task_criterion, recon_criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        cf_results = []
        truths = []

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
                text_missing_mask = batch_data['text_missing_mask'].to(device)
                audio_missing_mask = batch_data['audio_missing_mask'].to(device)
                vision_missing_mask = batch_data['vision_missing_mask'].to(device)
                text_lengths = batch_data['text_lengths'].to(device)
                cls_probs = batch_data['cls_probs'].to(device)
                text_cf = batch_data['text_cf'].to(device)
                binary_labels = binary_labels.view(-1).long()
                seven_labels = seven_labels.view(-1).long()
                cls_ids = 3 if task == 'binary' else 7
                cls_feats = []

                batch_size = text.size(0)

                for idx in range(cls_ids):
                    cls_feats.append(batch_data['cls_feats'][idx].to(device))

                net = nn.DataParallel(model) if hyp_params.distribute else model
                outputs = net(**dict(text=(text, text_m, None), audio=(audio, audio_m), vision=(vision, vision_m),
                                     input_masks=(text_mask, audio_mask, vision_mask),
                                     text_lengths=text_lengths, cls_feats=cls_feats, cls_probs=cls_probs,
                                     device=device, test=False, missing=False))

                outputs_m = net(**dict(text=(text, text_m, text_cf), audio=(audio, audio_m), vision=(vision, vision_m),
                                       input_masks=(text_mask, audio_mask, vision_mask),
                                       text_lengths=text_lengths, cls_feats=cls_feats, cls_probs=cls_probs,
                                       device=device, test=test, missing=True))

                if not test:
                    if task == 'binary':
                        loss_task_m = task_criterion(outputs_m['pred_m'], binary_labels)
                    else:  # seven
                        loss_task_m = task_criterion(outputs_m['pred_m'], seven_labels)

                    loss_joint_rep = 1 - F.cosine_similarity(outputs_m['joint_rep_m'], outputs['joint_rep'].detach(),
                                                             dim=-1).mean()

                    loss_attn = F.kl_div(F.log_softmax(outputs_m['attn_m'], dim=-1),
                                         F.softmax(outputs['attn'].detach(), dim=-1), reduction='batchmean')

                    mask = text_mask - text_missing_mask
                    loss_recon_text = recon_criterion(outputs_m['text_recon'], outputs['text_for_recon'], mask)
                    mask = audio_mask - audio_missing_mask
                    loss_recon_audio = recon_criterion(outputs_m['audio_recon'], audio, mask)
                    mask = vision_mask - vision_missing_mask
                    loss_recon_video = recon_criterion(outputs_m['vision_recon'], vision, mask)
                    loss_recon = loss_recon_text + loss_recon_audio + loss_recon_video

                    combined_loss = loss_task_m + \
                                    hyp_params.joint_rep_weight * loss_joint_rep + \
                                    hyp_params.attn_weight * loss_attn + \
                                    hyp_params.recon_weight * loss_recon
                    total_loss += combined_loss.item() * batch_size
                else:
                    # Collect the results into dictionary
                    results.append(outputs_m['pred_m'])
                    cf_results.append(outputs_m['cf_pred_m'])
                    if task == 'binary':
                        truths.append(binary_labels)
                    else:  # seven
                        truths.append(seven_labels)

        avg_loss = total_loss / (hyp_params.n_valid) if not test else None

        if test:
            results = torch.cat(results)
            cf_results = torch.cat(cf_results)
            truths = torch.cat(truths)
        else:
            results, cf_results, truths = None, None, None

        return avg_loss, results, cf_results, truths

    total_parameters = sum([param.nelement() for param in model.parameters()])
    bert_parameters = sum([param.nelement() for param in model.text_model.parameters()])
    print(f'Trainable Parameters: {total_parameters}...')
    print(f'BERT Parameters: {bert_parameters}...')
    print(f'CIDer Parameters: {total_parameters - bert_parameters}...')
    best_value = 0
    curr_patience = patience
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()

        train_loss = train(model, optimizer, task_criterion, recon_criterion)
        val_loss, results, cf_results, truths = evaluate(model, task_criterion, recon_criterion, test=False)
        evaluate(model, task_criterion, recon_criterion, test=True)

        end = time.time()
        duration = end - start

        print("-" * 50)
        print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f}'.format(
            epoch, duration, train_loss, val_loss))

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

        intermediate_value = ans[eval_metric]
        trial.report(intermediate_value, epoch)

        if intermediate_value > best_value:
            curr_patience = patience
            save_model(hyp_params, model, names={'model_name': hyp_params.model,
                                                 'trial_number': trial.number})
            best_value = intermediate_value
        else:
            curr_patience -= 1

        if curr_patience <= 0:
            break

    best_model = load_model(hyp_params, names={'model_name': hyp_params.model,
                                               'trial_number': trial.number})
    _, results, cf_results, truths = evaluate(best_model, task_criterion, recon_criterion, test=True)

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

    return ans[eval_metric]


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
