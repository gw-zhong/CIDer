import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder
# from modules.vanilla_transformer import TransformerEncoder
from modules.compressed_multihead_attention import CompressedMultiheadAttention
from transformers import BertTokenizer, BertModel

eps = 1e-12


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "tanh":
        return nn.Tanh
    else:
        raise NotImplementedError


class NaiveAttention(nn.Module):
    def __init__(self, dim, activation_fn='relu'):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            _get_activation_fn(activation_fn)(),
            nn.Linear(dim, 1)
        )

    def forward(self, inputs):
        """
        :param inputs: (B, T, D)
        :return: (B, D)
        """
        scores = self.attention(inputs)  # (B, T, 1)
        output = torch.matmul(torch.softmax(scores, dim=1).transpose(1, 2), inputs).squeeze(1)
        return output


class CIDERModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a CIDer model.
        """
        super(CIDERModel, self).__init__()
        self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.multimodal_layers = hyp_params.multimodal_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.language = hyp_params.language

        self.distribute = hyp_params.distribute
        self.aligned = hyp_params.aligned

        self.output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        self.l_len = self.orig_l_len
        self.a_len = self.orig_l_len  # Word-level alignment
        self.v_len = self.orig_l_len  # Word-level alignment
        # self.a_len = self.orig_a_len
        # self.v_len = self.orig_v_len

        # Prepare BERT model
        self.text_model = BertTextEncoder(language=hyp_params.language)

        # 1. Temporal convolutional blocks
        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=1, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=1, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=1, bias=False)

        # 2. Unimodal compressing blocks
        self.a_compress = CompressedMultiheadAttention(embed_dim=self.embed_dim,
                                                       num_heads=1)
        self.v_compress = CompressedMultiheadAttention(embed_dim=self.embed_dim,
                                                       num_heads=1)

        # 3. RNN encoder
        self.t = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)
        self.a = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)
        self.v = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)

        # 4. Multimodal fusion block
        self.multimodal_transformer = TransformerEncoder(embed_dim=self.embed_dim,
                                                         lens=(self.l_len, self.a_len),
                                                         num_heads=self.num_heads,
                                                         layers=self.multimodal_layers,
                                                         attn_dropout=self.attn_dropout,
                                                         embed_dropout=self.embed_dropout,
                                                         relu_dropout=self.relu_dropout,
                                                         res_dropout=self.res_dropout)

        # 5. Attention mechanism
        self.attn_l = NaiveAttention(self.embed_dim, activation_fn='tanh')
        self.attn_a = NaiveAttention(self.embed_dim, activation_fn='tanh')
        self.attn_v = NaiveAttention(self.embed_dim, activation_fn='tanh')

        # 6. Reconstruction decoder
        self.recon_text = nn.Linear(self.embed_dim, self.orig_d_l)
        self.recon_audio = nn.ModuleList(
            [nn.Linear(self.a_len, self.orig_a_len),
             nn.Linear(self.embed_dim, self.orig_d_a)]
        )
        self.recon_vision = nn.ModuleList(
            [nn.Linear(self.v_len, self.orig_v_len),
             nn.Linear(self.embed_dim, self.orig_d_v)]
        )

        # 7. Projection layers
        combined_dim = self.embed_dim * 3
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        # self.out_layer = nn.Linear(combined_dim, self.output_dim)  # original classifier

        # 8. Causal layers
        self.cls_encoders_l = nn.Sequential(
            nn.Linear(self.orig_d_l, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        if not hyp_params.cross_dataset:
            self.cls_encoders_a = nn.Sequential(
                nn.Linear(self.orig_d_a, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
            self.cls_encoders_v = nn.Sequential(
                nn.Linear(self.orig_d_v, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
        else:
            self.cls_encoders_a = nn.Sequential(
                nn.Linear(self.orig_d_a, self.embed_dim),
                nn.LayerNorm(self.embed_dim, elementwise_affine=False, bias=False),  # just add a LayerNorm due to the unstable non-linguistic features
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim, elementwise_affine=False, bias=False),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
            self.cls_encoders_v = nn.Sequential(
                nn.Linear(self.orig_d_v, self.embed_dim),
                nn.LayerNorm(self.embed_dim, elementwise_affine=False, bias=False),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim, elementwise_affine=False, bias=False),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
        self.classifiers = nn.ModuleList([])
        for i in range(self.output_dim):
            new_classifier = nn.Linear(combined_dim * 2, self.output_dim)
            self.classifiers.append(new_classifier)

    def forward_once(self, x_l, x_a, x_v, mask_t, mask_a, mask_v, t_lens, cls_feats, cls_probs, device, test, missing, counterfactual):
        """
        text should have dimension [batch_size, 3, seq_len]
        audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        if self.distribute:
            self.t.flatten_parameters()
            self.a.flatten_parameters()
            self.v.flatten_parameters()
        x_l = self.text_model(x_l, use_finetune=True)
        text_for_recon = x_l.detach()
        #################################################################################
        # Project the textual/visual/audio features & Compress the sequence length
        x_l = x_l.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        proj_x_l = self.proj_l(x_l)
        proj_x_a = self.proj_a(x_a)
        proj_x_v = self.proj_v(x_v)  # (bs, embed, seq)
        proj_x_l = proj_x_l.permute(2, 0, 1)  # (seq, bs, embed)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        #################################################################################
        # Compress nonverbal modalities (Cuz their tokens are too much
        if self.aligned:
            s_h_a = proj_x_a
            s_h_v = proj_x_v
        else:
            s_h_a, _ = self.a_compress(proj_x_l, proj_x_a, proj_x_a, mask_t, mask_a, t_lens)
            s_h_v, _ = self.v_compress(proj_x_l, proj_x_v, proj_x_v, mask_t, mask_v, t_lens)
        #################################################################################
        # Use RNN to encode
        h_l, _ = self.t(proj_x_l)
        h_a, _ = self.a(s_h_a)
        h_v, _ = self.v(s_h_v)
        #################################################################################
        # Multimodal fusion
        # Get total sequence and feed into Multimodal Transformer
        x = torch.cat((h_l, h_a, h_v), dim=0)
        x, attns = self.multimodal_transformer(x)

        h_l = x[:self.l_len]  # (seq, bs, embed)
        h_a = x[self.l_len:self.l_len + self.a_len]  # (seq, bs, embed)
        h_v = x[self.l_len + self.a_len:]  # (seq, bs, embed)

        g_l = self.attn_l(h_l.transpose(0, 1))
        g_a = self.attn_a(h_a.transpose(0, 1))
        g_v = self.attn_v(h_v.transpose(0, 1))

        last_hs = torch.cat((g_l, g_a, g_v), dim=-1)

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        last_rep = last_hs_proj
        # output = self.out_layer(last_rep)  # usual output
        # output = F.log_softmax(output, dim=-1)  # log probabilities for predictions
        #################################################################################
        # Causal intervention
        bz = last_rep.shape[0]
        output_probs = torch.zeros(bz, self.output_dim).to(device)  # causal output
        for i in range(self.output_dim):
            tmp_cls_feat = cls_feats[i]
            tmp_l = tmp_cls_feat[:, :self.orig_d_l]
            tmp_a = tmp_cls_feat[:, self.orig_d_l:self.orig_d_l + self.orig_d_a]
            tmp_v = tmp_cls_feat[:, self.orig_d_l + self.orig_d_a:]
            tmp_l = self.cls_encoders_l(tmp_l)
            tmp_a = self.cls_encoders_a(tmp_a)
            tmp_v = self.cls_encoders_v(tmp_v)
            tmp_causal = torch.concat((last_rep, tmp_l, tmp_a, tmp_v), dim=-1)
            tmp_logits = self.classifiers[i](tmp_causal)
            tmp_probs = F.softmax(tmp_logits, dim=-1)
            causal_probs = tmp_probs * cls_probs[:, [i]]
            output_probs += causal_probs
        #################################################################################
        # Log probabilities for predictions
        output = torch.log(output_probs)
        # Outputs
        suffix = '_m' if missing else ''
        prefix = 'cf_' if counterfactual else ''
        res = {
            f'{prefix}pred{suffix}': output,
            f'{prefix}joint_rep{suffix}': last_hs_proj,
            f'{prefix}attn{suffix}': attns[-1]
        }

        # Low-level feature reconstruction
        if not test:
            if missing:
                text_recon = self.recon_text(h_l.permute(1, 0, 2))
                audio_recon = self.recon_audio[1](F.relu(self.recon_audio[0](h_a.permute(1, 2, 0))).transpose(1, 2))
                vision_recon = self.recon_vision[1](F.relu(self.recon_vision[0](h_v.permute(1, 2, 0))).transpose(1, 2))
                res.update(
                    {
                        'text_recon': text_recon,
                        'audio_recon': audio_recon,
                        'vision_recon': vision_recon,
                    }
                )
            else:
                res.update({'text_for_recon': text_for_recon})

        return res

    def forward(self, text, audio, vision, input_masks, text_lengths, cls_feats, cls_probs, device, test, missing):
        text, text_m, text_cf = text
        audio, audio_m = audio
        vision, vision_m = vision
        text_mask, audio_mask, vision_mask = input_masks

        if not missing:
            # complete view
            res = self.forward_once(text, audio, vision,
                                    text_mask, audio_mask, vision_mask, text_lengths,
                                    cls_feats, cls_probs, device, test=test, missing=False, counterfactual=False)
        else:
            # incomplete view
            res = self.forward_once(text_m, audio_m, vision_m,
                                      text_mask, audio_mask, vision_mask, text_lengths,
                                      cls_feats, cls_probs, device, test=test, missing=True, counterfactual=False)
            if test:
                # counterfactual flow
                cf_res = self.forward_once(text_cf, audio_m, vision_m,
                                           text_mask, audio_mask, vision_mask, text_lengths,
                                           cls_feats, cls_probs, device, test=test, missing=True, counterfactual=True)
                res.update(**cf_res)

        return res


class BertTextEncoder(nn.Module):
    def __init__(self, language='en'):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('/data/zhonggw/pretrained_models/bert_en', do_lower_case=True)
            self.model = model_class.from_pretrained('/data/zhonggw/pretrained_models/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('/data/zhonggw/pretrained_models/bert_cn')
            self.model = model_class.from_pretrained('/data/zhonggw/pretrained_models/bert_cn')

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()

    def forward(self, text, use_finetune):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
