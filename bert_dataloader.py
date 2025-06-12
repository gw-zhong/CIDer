import joblib
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch


special_token_ids = [
    0,    # [PAD]
    100,  # [UNK]
    101,  # [CLS]
    102,  # [SEP]
    103]  # [MASK]
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        self.cls_feats = {}
        cls_id = 3 if args.task == 'binary' else 7
        version = 'aligned' if args.aligned else 'unaligned'
        data_distribution = 'ood' if args.ood else 'iid'
        self.data_distribution = data_distribution
        cross_dataset_status = '[cross_dataset]' if args.cross_dataset else ''
        self.cross_dataset_status = cross_dataset_status
        if mode == 'train':
            self.cls_probs = np.load(f'./{args.dataset}_probs/{self.data_distribution}/{args.task}/{cross_dataset_status}{args.dataset}_{mode}_{args.task}.npy').astype(np.float32)
        else:
            self.cls_probs = np.array([1/cls_id for i in range(cls_id)]).astype(np.float32)
        for idx in range(cls_id):
            self.cls_feats[idx] = np.load(f'./{args.dataset}_feats/{self.data_distribution}/{args.task}/{cross_dataset_status}{args.dataset}_{mode}_{args.task}_{version}_{idx}.npy').astype(np.float32)
        if mode != 'train':
            all_feats = [self.cls_feats[idx] for idx in range(cls_id)]
            all_mean = np.mean(all_feats, axis=0)
            for idx in range(cls_id):
                self.cls_feats[idx] = all_mean
        if args.cross_dataset:
            if args.dataset == 'mosi':
                self.test_set = 'mosei'
            else:
                self.test_set = 'mosi'
        else:
            self.test_set = args.dataset
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei
        }
        DATA_MAP[args.dataset]()

    def __init_mosi(self):
        if self.args.ood:
            if self.args.aligned:
                path = os.path.join(self.args.data_path, 'OOD_' + str.upper(self.args.dataset), self.args.task, str.upper(self.args.dataset) + '_aligned_50.pkl')
            else:
                path = os.path.join(self.args.data_path, 'OOD_' + str.upper(self.args.dataset), self.args.task, str.upper(self.args.dataset) + '_unaligned_50.pkl')
            with open(path, 'rb') as f:
                data = joblib.load(f)
        else:
            if self.args.aligned:
                if self.args.cross_dataset:
                    if self.mode == 'test':
                        path = os.path.join(self.args.data_path, 'MSA_cross_datasets', self.test_set + '.pkl')
                    else:
                        path = os.path.join(self.args.data_path, 'MSA_cross_datasets', self.args.dataset + '.pkl')
                else:
                    path = os.path.join(self.args.data_path, 'CMU-' + str.upper(self.args.dataset), 'Processed', 'aligned_50.pkl')
            else:
                path = os.path.join(self.args.data_path, 'CMU-' + str.upper(self.args.dataset), 'Processed', 'unaligned_50.pkl')
            with open(path, 'rb') as f:
                data = pickle.load(f)
        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        seven = np.round(np.clip(data[self.mode]['regression_labels'], a_min=-3., a_max=3.)) + 3  # [0, 1, 2 ,3 ,4 ,5, 6]
        if self.args.cross_dataset:
            binary = np.where(data[self.mode]['regression_labels'] < 0, 0, np.where(data[self.mode]['regression_labels'] == 0, 1, 2))
            self.labels = {
                'M': data[self.mode]['regression_labels'].astype(np.float32),
                'binary': binary.astype(np.int64),  # [0, 1, 2]
                'seven': seven.astype(np.int64)
            }
        else:
            self.labels = {
                'M': data[self.mode]['regression_labels'].astype(np.float32),
                'binary': data[self.mode]['classification_labels'].astype(np.int64),  # [0, 1, 2]
                'seven': seven.astype(np.int64)
            }
        self.text_lengths = np.sum(self.text[:, 1], axis=1).astype(np.int16).tolist()

        if not self.args.aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        else:
            self.audio_lengths, self.vision_lengths = self.text_lengths, self.text_lengths
        self.audio[self.audio == -np.inf] = 0

        if self.mode == 'train':
            if os.path.exists(f'./{self.cross_dataset_status}{self.args.dataset}_bias_{self.args.bias_thresh}_{self.data_distribution}_{self.args.task}_words_dict.pkl'):
                print('Already has a bias word dict....')
            else:
                print('There is not a bias word dict, making the dict....')
                words_cnt_dict = {}
                for idx, input_ids in enumerate(self.text[:, 0, :]):
                    for input_id in input_ids:
                        if input_id in special_token_ids: continue
                        if input_id not in words_cnt_dict:
                            words_cnt_dict[input_id] = {}
                            words_cnt_dict[input_id]['sum'] = 0
                            if self.args.task == 'binary':
                                for i in range(3):
                                    words_cnt_dict[input_id][i] = 0
                            else:
                                for i in range(7):
                                    words_cnt_dict[input_id][i] = 0
                        words_cnt_dict[input_id][self.labels[self.args.task][idx]] += 1
                        words_cnt_dict[input_id]['sum'] += 1
                bias_words_dict = {}
                for input_id, sub_dict in words_cnt_dict.items():
                    if self.args.task == 'binary':
                        if sub_dict[0] > 0 or sub_dict[2] > 0:
                            std = np.std(np.array([sub_dict[0], sub_dict[2]], dtype=np.float32))
                            mean = np.mean(np.array([sub_dict[0], sub_dict[2]], dtype=np.float32))
                    else:
                        std = np.std(np.array([sub_dict[0], sub_dict[1], sub_dict[2], sub_dict[3], sub_dict[4], sub_dict[5], sub_dict[6]], dtype=np.float32))
                        mean = np.mean(np.array([sub_dict[0], sub_dict[1], sub_dict[2], sub_dict[3], sub_dict[4], sub_dict[5], sub_dict[6]], dtype=np.float32))
                    cv = std / mean
                    if cv >= 0.1:
                        if input_id not in bias_words_dict:
                            bias_words_dict[input_id] = {}
                        bias_words_dict[input_id]['sum'] = words_cnt_dict[input_id]['sum']
                        bias_words_dict[input_id]['cv'] = cv
                bias_words_dict = dict(sorted(bias_words_dict.items(), key=lambda x: x[1]['sum'], reverse=True)[:self.args.bias_thresh])
                with open(f'./{self.cross_dataset_status}{self.args.dataset}_bias_{self.args.bias_thresh}_{self.data_distribution}_{self.args.task}_words_dict.pkl', 'wb') as f:
                    pickle.dump(bias_words_dict, f)
        else:
            modalities_m, input_lens, input_masks, missing_masks = self.generate_m(
                (self.text[:, 0, :], self.audio, self.vision), self.text[:, 1, :],
                (self.text_lengths, self.audio_lengths, self.vision_lengths),
                missing_rate=self.args.missing_rate, missing_seed=self.args.seed, mode=self.args.missing_mode
            )
            self.text_m, self.audio_m, self.vision_m = modalities_m
            self.text_mask, self.audio_mask, self.vision_mask = input_masks
            self.text_missing_mask, self.audio_missing_mask, self.vision_missing_mask = missing_masks

            self.text_cf = self.generate_cf(self.text_m)
            Input_ids_cf = np.expand_dims(self.text_cf, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:, 2, :], 1)
            self.text_cf = np.concatenate((Input_ids_cf, Input_mask, Segment_ids), axis=1)

            Input_ids_m = np.expand_dims(self.text_m, 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

    def __init_mosei(self):
        return self.__init_mosi()

    def generate_m(self, modalities, text_input_mask, input_lens, missing_rate, missing_seed, mode='RMFM'):
        text, audio, vision = modalities
        text_lengths, audio_lengths, vision_lengths = input_lens
        text_input_len = np.argmin(np.concatenate((text_input_mask, np.zeros((text_input_mask.shape[0], 1))), axis=1),
                                   axis=1)  # 防止mask全一导致长度为0
        if missing_seed is not None:
            np.random.seed(missing_seed)
        audio_input_mask = np.array(
            [np.array([1] * length + [0] * (audio.shape[1] - length)) for length in audio_lengths])
        vision_input_mask = np.array(
            [np.array([1] * length + [0] * (vision.shape[1] - length)) for length in vision_lengths])
        if mode == 'RMFM':  # Random Modality Feature Missing (**different missing rates** for each modality's feature)
            input_mask = np.concatenate([text_input_mask, audio_input_mask, vision_input_mask], axis=1)
            if missing_rate is None:  # train mode
                random_missing_rate = np.random.rand(input_mask.shape[0], input_mask.shape[1])
                missing_mask = (np.random.uniform(size=input_mask.shape) > random_missing_rate) * input_mask
            else:
                missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask

            assert missing_mask.shape == input_mask.shape

            # [CLS] [SEP] Token unchanged.
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[text_input_len[i] - 1] = 1

            text_seq_len = text.shape[1]
            audio_seq_len = audio.shape[1]
            vision_seq_len = vision.shape[1]
            text_missing_mask = missing_mask[:, :text_seq_len]
            audio_missing_mask = missing_mask[:, text_seq_len:text_seq_len + audio_seq_len]
            vision_missing_mask = missing_mask[:, text_seq_len + audio_seq_len:]
            text_m = text_missing_mask * text + (100 * np.ones_like(text)) * (
                        text_input_mask - text_missing_mask)  # UNK token: 100.
            audio_m = audio_missing_mask.reshape(audio.shape[0], audio.shape[1], 1) * audio
            vision_m = vision_missing_mask.reshape(vision.shape[0], vision.shape[1], 1) * vision
        elif mode == 'RMM':  # Random Modality Missing (like TATE/EMMR/GCNet, etc.)
            modality_masks = np.ones((text.shape[0], 3))  # (Text, Audio, Video)
            missing_masks = (np.random.uniform(size=modality_masks.shape) > missing_rate) * modality_masks
            text_missing_mask = text_input_mask.copy()
            audio_missing_mask = audio_input_mask.copy()
            vision_missing_mask = vision_input_mask.copy()
            for i, instance in enumerate(missing_masks):
                if instance[0] == 0:  # missing Text
                    text_missing_mask[i] *= 0
                    text_missing_mask[i][0] = text_missing_mask[i][text_input_len[i] - 1] = 1
                if instance[1] == 0:  # missing Audio
                    audio_missing_mask[i] *= 0
                if instance[2] == 0:  # missing Video
                    vision_missing_mask[i] *= 0
            text_m = text_missing_mask * text + (100 * np.ones_like(text)) * (
                    text_input_mask - text_missing_mask)  # UNK token: 100.
            audio_m = audio_missing_mask.reshape(audio.shape[0], audio.shape[1], 1) * audio
            vision_m = vision_missing_mask.reshape(vision.shape[0], vision.shape[1], 1) * vision
        elif mode == 'TMFM':  # Temporal Modality Feature Missing (details in NIAT)
            assert self.args.aligned, 'Temporal Modality Feature Missing only supports aligned data'
            missing_mask = (np.random.uniform(size=text_input_mask.shape) > missing_rate) * text_input_mask
            text_missing_mask = audio_missing_mask = vision_missing_mask = missing_mask.copy()

            assert text_missing_mask.shape == text_input_mask.shape
            assert audio_missing_mask.shape == audio_input_mask.shape
            assert vision_missing_mask.shape == vision_input_mask.shape

            # [CLS] [SEP] Token unchanged.
            for i, instance in enumerate(text_missing_mask):
                instance[0] = instance[text_input_len[i] - 1] = 1

            text_m = text_missing_mask * text + (100 * np.ones_like(text)) * (
                    text_input_mask - text_missing_mask)  # UNK token: 100.
            audio_m = audio_missing_mask.reshape(audio.shape[0], audio.shape[1], 1) * audio
            vision_m = vision_missing_mask.reshape(vision.shape[0], vision.shape[1], 1) * vision
        elif mode == 'STMFM':  # Structural Temporal Modality Feature Missing (details in NIAT)
            assert self.args.aligned, 'Structural Temporal Modality Feature Missing only supports aligned data'
            missing_block_len = np.around((text_input_len - 2) * missing_rate).astype(np.int32)
            missing_mask = text_input_mask.copy()
            for i, instance in enumerate(missing_mask):
                start_p = np.random.randint(low=1, high=text_input_len[i] - missing_block_len[i])
                missing_mask[i, start_p:start_p + missing_block_len[i]] = 0
            text_missing_mask = audio_missing_mask = vision_missing_mask = missing_mask.copy()

            assert text_missing_mask.shape == text_input_mask.shape
            assert audio_missing_mask.shape == audio_input_mask.shape
            assert vision_missing_mask.shape == vision_input_mask.shape

            # [CLS] [SEP] Token unchanged.
            for i, instance in enumerate(text_missing_mask):
                instance[0] = instance[text_input_len[i] - 1] = 1

            text_m = text_missing_mask * text + (100 * np.ones_like(text)) * (
                    text_input_mask - text_missing_mask)  # UNK token: 100.
            audio_m = audio_missing_mask.reshape(audio.shape[0], audio.shape[1], 1) * audio
            vision_m = vision_missing_mask.reshape(vision.shape[0], vision.shape[1], 1) * vision
        elif mode == 'RMFM_same':  # Random Modality Feature Missing with **same missing rates** (like TFR-Net/NIAT/EMT-DLFR, etc.)
            text_missing_mask = (np.random.uniform(size=text_input_mask.shape) > missing_rate[0]) * text_input_mask
            audio_missing_mask = (np.random.uniform(size=audio_input_mask.shape) > missing_rate[1]) * audio_input_mask
            vision_missing_mask = (np.random.uniform(size=vision_input_mask.shape) > missing_rate[2]) * vision_input_mask
            # Specific Modality Missing is included in this case

            assert text_missing_mask.shape == text_input_mask.shape
            assert audio_missing_mask.shape == audio_input_mask.shape
            assert vision_missing_mask.shape == vision_input_mask.shape

            # [CLS] [SEP] Token unchanged.
            for i, instance in enumerate(text_missing_mask):
                instance[0] = instance[text_input_len[i] - 1] = 1

            text_m = text_missing_mask * text + (100 * np.ones_like(text)) * (
                        text_input_mask - text_missing_mask)  # UNK token: 100.
            audio_m = audio_missing_mask.reshape(audio.shape[0], audio.shape[1], 1) * audio
            vision_m = vision_missing_mask.reshape(vision.shape[0], vision.shape[1], 1) * vision
        else:
            raise ValueError('Unknown missing mode...')

        modalities_m = (text_m, audio_m, vision_m)
        input_lens = (text_input_len, audio_lengths, vision_lengths)
        input_masks = (text_input_mask, audio_input_mask, vision_input_mask)
        missing_masks = (text_missing_mask, audio_missing_mask, vision_missing_mask)
        return modalities_m, input_lens, input_masks, missing_masks
    
    def generate_cf(self, text):
        print(f'Loading existing bias word dict for {self.mode} set....')
        with open(f'./{self.cross_dataset_status}{self.args.dataset}_bias_{self.args.bias_thresh}_{self.data_distribution}_{self.args.task}_words_dict.pkl', 'rb') as f:
                bias_words_dict = pickle.load(f)

        text_cf = []
        for input_ids in text:
            input_ids_cf = []
            for input_id in input_ids:
                if input_id in special_token_ids:
                    input_ids_cf.append(input_id)
                else:
                    if input_id not in bias_words_dict:
                        input_ids_cf.append(103)  # [MASK] token is 103
                    else:
                        input_ids_cf.append(input_id)
            text_cf.append(np.array(input_ids_cf)[np.newaxis, :])
        text_cf = np.concatenate(text_cf, axis=0)
        return text_cf

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if ((instance[index] == padding).all()):
                        if (index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index + 20])
                            break
                    else:
                        truncated_feature.append(instance[index:index + 20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature

        text_length, audio_length, video_length = self.args.seq_lens
        self.vision = Truncated(self.vision, video_length)
        self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        if self.mode == 'train':
            modalities_m, input_lens, input_masks, missing_masks = self.generate_m(
                (self.text[index][np.newaxis, :, :][:, 0, :],
                 self.audio[index][np.newaxis, :, :],
                 self.vision[index][np.newaxis, :, :]),
                self.text[index][np.newaxis, :, :][:, 1, :],
                ([self.text_lengths[index]],
                 [self.audio_lengths[index]],
                 [self.vision_lengths[index]]),
                missing_rate=None, missing_seed=None)
            self.text_m, self.audio_m, self.vision_m = modalities_m
            self.text_mask, self.audio_mask, self.vision_mask = input_masks
            self.text_missing_mask, self.audio_missing_mask, self.vision_missing_mask = missing_masks
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[index][np.newaxis, :, :][:, 2, :], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1).squeeze()
            self.audio_m = self.audio_m.squeeze()
            self.vision_m = self.vision_m.squeeze()
            self.text_mask = self.text_mask.squeeze()
            self.audio_mask = self.audio_mask.squeeze()
            self.vision_mask = self.vision_mask.squeeze()
            self.text_missing_mask = self.text_missing_mask.squeeze()
            self.audio_missing_mask = self.audio_missing_mask.squeeze()
            self.vision_missing_mask = self.vision_missing_mask.squeeze()
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]),  # [3, 50]
                'text_m': torch.Tensor(self.text_m),  # [3, 50]
                'text_lengths': self.text_lengths[index],
                'text_mask': self.text_mask,
                'text_missing_mask': torch.Tensor(self.text_missing_mask),
                'audio': torch.Tensor(self.audio[index]),
                'audio_m': torch.Tensor(self.audio_m),
                'audio_lengths': self.audio_lengths[index],
                'audio_mask': self.audio_mask,
                'audio_missing_mask': self.audio_missing_mask,
                'vision': torch.Tensor(self.vision[index]),
                'vision_m': torch.Tensor(self.vision_m),
                'vision_lengths': self.vision_lengths[index],
                'vision_mask': self.vision_mask,
                'vision_missing_mask': self.vision_missing_mask,
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
                'cls_feats': {k: torch.Tensor(v) for k, v in self.cls_feats.items()},
                'cls_probs': torch.Tensor(self.cls_probs),
            }
        else:
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]),  # [3, 50]
                'text_m': torch.Tensor(self.text_m[index]),  # [3, 50]
                'text_lengths': self.text_lengths[index],
                'text_mask': self.text_mask[index],
                'text_missing_mask': torch.Tensor(self.text_missing_mask[index]),
                'audio': torch.Tensor(self.audio[index]),
                'audio_m': torch.Tensor(self.audio_m[index]),
                'audio_lengths': self.audio_lengths[index],
                'audio_mask': self.audio_mask[index],
                'audio_missing_mask': self.audio_missing_mask[index],
                'vision': torch.Tensor(self.vision[index]),
                'vision_m': torch.Tensor(self.vision_m[index]),
                'vision_lengths': self.vision_lengths[index],
                'vision_mask': self.vision_mask[index],
                'vision_missing_mask': self.vision_missing_mask[index],
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
                'cls_feats': {k: torch.Tensor(v) for k, v in self.cls_feats.items()},
                'cls_probs': torch.Tensor(self.cls_probs),
                'text_cf': torch.Tensor(self.text_cf[index]),
            }
        return sample
