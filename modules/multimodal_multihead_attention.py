import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

# Code adapted from the fairseq repo.


class MultimodalMultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, lens, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.l_len, self.a_len = lens
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.final_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(7)])

        self.s_l = nn.Linear(4 * embed_dim, embed_dim)
        self.s_a = nn.Linear(4 * embed_dim, embed_dim)
        self.s_v = nn.Linear(4 * embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        for i in range(7):
            nn.init.xavier_uniform_(self.out_proj[i].weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            for i in range(7):
                nn.init.constant_(self.out_proj[i].bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        #################################################################################
        # Local-to-Local Interaction
        # SA[L, A, V]
        attn_weights_lav = attn_weights.clone()
        v_lav = v.clone()
        attn_weights_lav = F.softmax(attn_weights_lav.float(), dim=-1).type_as(attn_weights_lav)
        attn_weights_lav = F.dropout(attn_weights_lav, p=self.attn_dropout, training=self.training)
        attn_lav = torch.bmm(attn_weights_lav, v_lav)
        assert list(attn_lav.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_lav = attn_lav.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_lav = self.out_proj[0](attn_lav)
        # SA[L, A]
        attn_weights_la = attn_weights.clone()
        v_la = v.clone()
        mask = torch.ones(v.shape[0], v.shape[1]).to(attn_weights.device)
        mask[:, self.l_len + self.a_len:] = 0
        mask1 = mask.unsqueeze(-1).expand(v_la.shape[0], v_la.shape[1], v_la.shape[2])
        v_la = v_la * mask1
        mask_la = torch.zeros_like(attn_weights_la)
        mask_la[:, :self.l_len + self.a_len, self.l_len + self.a_len:] = -1e9
        mask_la[:, self.l_len + self.a_len:, :] = -1e9
        attn_weights_la = attn_weights_la + mask_la
        attn_weights_la = F.softmax(attn_weights_la.float(), dim=-1).type_as(attn_weights_la)
        mask2 = mask.unsqueeze(-1).expand(attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2])
        attn_weights_la = attn_weights_la * mask2
        attn_la = torch.bmm(attn_weights_la, v_la)
        assert list(attn_la.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_la = attn_la.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_la = self.out_proj[1](attn_la)
        # SA[L, V]
        attn_weights_lv = attn_weights.clone()
        v_lv = v.clone()
        mask = torch.ones(v.shape[0], v.shape[1]).to(attn_weights.device)
        mask[:, self.l_len:self.l_len + self.a_len] = 0
        mask1 = mask.unsqueeze(-1).expand(v_la.shape[0], v_la.shape[1], v_la.shape[2])
        v_lv = v_lv * mask1
        mask_lv = torch.zeros_like(attn_weights_lv)
        mask_lv[:, :self.l_len, self.l_len:self.l_len + self.a_len] = -1e9
        mask_lv[:, self.l_len:self.l_len + self.a_len, :] = -1e9
        mask_lv[:, self.l_len + self.a_len:, self.l_len:self.l_len + self.a_len] = -1e9
        attn_weights_lv = attn_weights_lv + mask_lv
        attn_weights_lv = F.softmax(attn_weights_lv.float(), dim=-1).type_as(attn_weights_lv)
        mask2 = mask.unsqueeze(-1).expand(attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2])
        attn_weights_lv = attn_weights_lv * mask2
        attn_lv = torch.bmm(attn_weights_lv, v_lv)
        assert list(attn_lv.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_lv = attn_lv.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_lv = self.out_proj[2](attn_lv)
        # SA[A, V]
        attn_weights_av = attn_weights.clone()
        v_av = v.clone()
        mask = torch.ones(v.shape[0], v.shape[1]).to(attn_weights.device)
        mask[:, :self.l_len] = 0
        mask1 = mask.unsqueeze(-1).expand(v_la.shape[0], v_la.shape[1], v_la.shape[2])
        v_av = v_av * mask1
        mask_av = torch.zeros_like(attn_weights_av)
        mask_av[:, :self.l_len, :] = -1e9
        mask_av[:, self.l_len:, :self.l_len] = -1e9
        attn_weights_av = attn_weights_av + mask_av
        attn_weights_av = F.softmax(attn_weights_av.float(), dim=-1).type_as(attn_weights_av)
        mask2 = mask.unsqueeze(-1).expand(attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2])
        attn_weights_av = attn_weights_av * mask2
        attn_av = torch.bmm(attn_weights_av, v_av)
        assert list(attn_av.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_av = attn_av.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_av = self.out_proj[3](attn_av)
        # SA[L]
        attn_weights_l = attn_weights.clone()
        v_l = v.clone()
        mask = torch.ones(v.shape[0], v.shape[1]).to(attn_weights.device)
        mask[:, self.l_len:] = 0
        mask1 = mask.unsqueeze(-1).expand(v_la.shape[0], v_la.shape[1], v_la.shape[2])
        v_l = v_l * mask1
        mask_l = torch.zeros_like(attn_weights_l)
        mask_l[:, :self.l_len, self.l_len:] = -1e9
        mask_l[:, self.l_len:, :] = -1e9
        attn_weights_l = attn_weights_l + mask_l
        attn_weights_l = F.softmax(attn_weights_l.float(), dim=-1).type_as(attn_weights_l)
        mask2 = mask.unsqueeze(-1).expand(attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2])
        attn_weights_l = attn_weights_l * mask2
        attn_l = torch.bmm(attn_weights_l, v_l)
        assert list(attn_l.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_l = attn_l.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_l = self.out_proj[4](attn_l)
        # SA[A]
        attn_weights_a = attn_weights.clone()
        v_a = v.clone()
        mask = torch.ones(v.shape[0], v.shape[1]).to(attn_weights.device)
        mask[:, :self.l_len] = 0
        mask[:, self.l_len + self.a_len:] = 0
        mask1 = mask.unsqueeze(-1).expand(v_la.shape[0], v_la.shape[1], v_la.shape[2])
        v_a = v_a * mask1
        mask_a = torch.zeros_like(attn_weights_a)
        mask_a[:, :self.l_len, :] = -1e9
        mask_a[:, self.l_len:self.l_len + self.a_len, :self.l_len] = -1e9
        mask_a[:, self.l_len:self.l_len + self.a_len, self.l_len + self.a_len:] = -1e9
        mask_a[:, self.l_len + self.a_len:, :] = -1e9
        attn_weights_a = attn_weights_a + mask_a
        attn_weights_a = F.softmax(attn_weights_a.float(), dim=-1).type_as(attn_weights_a)
        mask2 = mask.unsqueeze(-1).expand(attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2])
        attn_weights_a = attn_weights_a * mask2
        attn_a = torch.bmm(attn_weights_a, v_a)
        assert list(attn_a.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_a = attn_a.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_a = self.out_proj[5](attn_a)
        # SA[V]
        attn_weights_v = attn_weights.clone()
        v_v = v.clone()
        mask = torch.ones(v.shape[0], v.shape[1]).to(attn_weights.device)
        mask[:, :self.l_len + self.a_len] = 0
        mask1 = mask.unsqueeze(-1).expand(v_la.shape[0], v_la.shape[1], v_la.shape[2])
        v_v = v_v * mask1
        mask_v = torch.zeros_like(attn_weights_v)
        mask_v[:, :self.l_len + self.a_len, :] = -1e9
        mask_v[:, self.l_len + self.a_len:, :self.l_len + self.a_len] = -1e9
        attn_weights_v = attn_weights_v + mask_v
        attn_weights_v = F.softmax(attn_weights_v.float(), dim=-1).type_as(attn_weights_v)
        mask2 = mask.unsqueeze(-1).expand(attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2])
        attn_weights_v = attn_weights_v * mask2
        attn_v = torch.bmm(attn_weights_v, v_v)
        assert list(attn_v.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_v = attn_v.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_v = self.out_proj[6](attn_v)
        # Summarization Network
        text = torch.cat(
            (attn_lav[:self.l_len], attn_la[:self.l_len], attn_lv[:self.l_len], attn_l[:self.l_len]), dim=-1)
        audio = torch.cat(
            (attn_lav[self.l_len:self.l_len + self.a_len], attn_la[self.l_len:self.l_len + self.a_len],
            attn_av[self.l_len:self.l_len + self.a_len], attn_a[self.l_len:self.l_len + self.a_len]), dim=-1)
        vision = torch.cat(
            (attn_lav[self.l_len + self.a_len:], attn_lv[self.l_len + self.a_len:],
             attn_av[self.l_len + self.a_len:], attn_v[self.l_len + self.a_len:]), dim=-1)
        text = self.s_l(text)
        audio = self.s_a(audio)
        vision = self.s_v(vision)
        final_attn = torch.cat((text, audio, vision), dim=0)
        final_attn = self.final_out_proj(final_attn)
        #################################################################################

        return final_attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self .embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
