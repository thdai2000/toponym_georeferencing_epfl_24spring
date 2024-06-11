import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from GrouperData import GrouperDataset
from english_encoding import *
from tqdm import tqdm

# Set up TensorBoard logging
log_dir = './logs'
writer = SummaryWriter(log_dir)


import torch
import torch.nn as nn
from torch.nn import Transformer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PointerNetwork(nn.Module):
    def __init__(self, n_hidden: int):
        super().__init__()
        self.n_hidden = n_hidden
        self.w1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.w2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.v = nn.Linear(n_hidden, 1, bias=False)

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:

        # (B, Nd, Ne, C) <- (B, Ne, C)
        encoder_transform = self.w1(x_encoder).unsqueeze(1).expand(
          -1, x_decoder.shape[1], -1, -1)
        # (B, Nd, 1, C) <- (B, Nd, C)
        decoder_transform = self.w2(x_decoder).unsqueeze(2)
        # (B, Nd, Ne) <- (B, Nd, Ne, C), (B, Nd, 1, C)
        prod = self.v(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        # (B, Nd, Ne) <- (B, Nd, Ne)
        log_score = torch.nn.functional.log_softmax(prod, dim=-1)
        return log_score


class GrouperTransformerEncoderDecoderAttentionNoFont(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=64, nhead=2, num_encoder_layers=1, num_decoder_layers=1):
        super(GrouperTransformerEncoderDecoderAttentionNoFont, self).__init__()

        self.type = 'GrouperTransformerEncoderDecoderAttentionNoFont'
        self.pre_project = nn.Linear(feature_dim+2, hidden_dim, bias=False)  # shared across encoder and decoder
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.pos_decoder = PositionalEncoding(hidden_dim*2)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                        dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        # self.post_project = nn.Linear(hidden_dim, hidden_dim*2)
        self.reduce_decoder_input_dim = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead,
                                                        dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.special_token_id = {'<sot>': 16, '<eot>': 17, '<pad>': 18}
        # self.output_layer = nn.Linear(hidden_dim*2, 18)
        self.pointer = PointerNetwork(hidden_dim)

        print('{} loaded.'.format(self.type))

    def generate_attention_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        # mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

    def generate_padding_mask(self, tgt_ids, tgt_len):
        tgt_key_padding_mask = torch.zeros_like(tgt_ids, dtype=torch.bool)
        for i in range(tgt_len.shape[0]):
            tgt_key_padding_mask[i, tgt_len[i] + 1:] = True
        return tgt_key_padding_mask

    def forward(self, src, query_id, tgt_ids=None, tgt_len=None):

        # src: (N,L,f_sz)  src: (N,L+3,f_sz+2)
        # append special tokens features to input, <sot> and <eot> are binary vectors
        src_padded = torch.zeros((src.shape[0], src.shape[1] + 2, src.shape[2] + 2), dtype=torch.float).to(src.device)
        src_padded[:, src.shape[1], src.shape[2]] = 1.0  # add <sot> vector
        src_padded[:, src.shape[1] + 1, src.shape[2] + 1] = 1.0  # add <eot> vector
        src_padded[:, :src.shape[1], :src.shape[2]] = src

        src_padded_emb = self.pre_project(src_padded)
        # src_padded_emb = self.pos_encoder(src_padded_emb)

        # src_padding_mask = torch.zeros((src_padded_emb.shape[0], src_padded_emb.shape[1]), dtype=torch.bool)
        # src_padding_mask[:, -1] = True
        # src_padding_mask = src_padding_mask.to(src.device)
        memory = self.transformer_encoder(src_padded_emb)
        # memory = self.post_project(memory)

        # src_padded: (N,L+3,h_dim)  src,memory: (N,L+2,h_dim)  query_id: (N,1)  tgt_ids: (N,L+2)
        # training by teacher forcing
        if tgt_ids is not None:
            batch_indices = torch.arange(tgt_ids.shape[0]).unsqueeze(1).expand(tgt_ids.shape[0], tgt_ids.shape[1])
            # temporarily append one zero vector to facilitate tensor extraction
            memory_padded = torch.zeros((memory.shape[0], memory.shape[1]+1, memory.shape[2]), dtype=torch.float).to(src.device)
            memory_padded[:, :-1, :] = memory
            tgt_tensors = memory_padded[batch_indices, tgt_ids]
            query_tensors = memory[torch.arange(src.shape[0]), query_id].unsqueeze(1).expand(-1, tgt_tensors.shape[1], -1)
            decoder_input_cat = torch.cat([tgt_tensors, query_tensors], dim=-1)
            decoder_input = self.reduce_decoder_input_dim(decoder_input_cat)
            # decoder_input = self.pos_decoder(decoder_input)

            tgt_key_padding_mask = self.generate_padding_mask(tgt_ids, tgt_len).to(src.device)
            # tgt_mask = self.generate_attention_mask(decoder_input.shape[1]).to(src.device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

            decoder_hidden = self.transformer_decoder(decoder_input,
                                                      memory,
                                                      # memory_key_padding_mask=src_padding_mask,
                                                      tgt_mask=tgt_mask,
                                                      tgt_key_padding_mask=tgt_key_padding_mask)

            # decoder_output = self.output_layer(decoder_hidden)

            log_pointer_scores = self.pointer(decoder_hidden, memory)

        # inference or training without teacher forcing
        else:
            query_tensors = memory[torch.arange(memory.shape[0]), query_id].unsqueeze(1)

            decoder_input_cat = torch.cat([memory[:, 16, :].unsqueeze(1), query_tensors], dim=-1)  # <sot; query>

            decoder_input = self.reduce_decoder_input_dim(decoder_input_cat)
            # decoder_input = self.pos_decoder(decoder_input)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

            log_pointer_scores = torch.zeros((memory.shape[0], memory.shape[1], memory.shape[1]), dtype=torch.float).to(src.device)

            for t in range(memory.shape[1]-1):

                decoder_hidden = self.transformer_decoder(decoder_input,
                                                          memory,
                                                          # memory_key_padding_mask=src_padding_mask,
                                                          tgt_mask=tgt_mask)
                log_pointer_score = self.pointer(decoder_hidden, memory)
                log_pointer_scores[:, t, :] = log_pointer_score[:, t, :]

                next_id = torch.argmax(log_pointer_score[:, t, :], dim=-1)
                next_input = memory[torch.arange(memory.shape[0]), next_id]
                if t == memory.shape[1]-2:
                    break
                else:
                    next_input_cat = torch.cat([next_input.unsqueeze(1), query_tensors], dim=-1)
                    decoder_input = torch.cat([decoder_input, self.reduce_decoder_input_dim(next_input_cat)], dim=1)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

        return log_pointer_scores

class GrouperTransformerEncoderDecoderAttentionWithFont(nn.Module):
    def __init__(self, feature_dim=16, font_dim=24, hidden_dim=64, nhead=2, num_encoder_layers=1, num_decoder_layers=1):
        super(GrouperTransformerEncoderDecoderAttentionWithFont, self).__init__()

        self.type = 'GrouperTransformerEncoderDecoderAttentionWithFont'
        self.pre_project = nn.Linear(feature_dim, hidden_dim, bias=False)  # shared across encoder and decoder
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.pos_decoder = PositionalEncoding(hidden_dim*2)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim+font_dim+2, nhead=nhead,
                                                        dim_feedforward=(hidden_dim+font_dim+2)*2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        # self.post_project = nn.Linear(hidden_dim, hidden_dim*2)
        self.reduce_decoder_input_dim = nn.Linear((hidden_dim+font_dim+2)*2, hidden_dim+font_dim+2, bias=False)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim+font_dim+2, nhead=nhead,
                                                        dim_feedforward=(hidden_dim+font_dim+2)*2, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.special_token_id = {'<sot>': 16, '<eot>': 17, '<pad>': 18}
        # self.output_layer = nn.Linear(hidden_dim*2, 18)
        self.pointer = PointerNetwork(hidden_dim+font_dim+2)

        print('{} loaded.'.format(self.type))

    def generate_attention_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        # mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

    def generate_padding_mask(self, tgt_ids, tgt_len):
        tgt_key_padding_mask = torch.zeros_like(tgt_ids, dtype=torch.bool)
        for i in range(tgt_len.shape[0]):
            tgt_key_padding_mask[i, tgt_len[i] + 1:] = True
        return tgt_key_padding_mask

    def forward(self, src, font, query_id, tgt_ids=None, tgt_len=None):

        # src: (N,L,f_sz)  font: (N,L,128), src: (N,L+2,f_sz+2)
        # append special tokens features to input, <sot> and <eot> are binary vectors


        src_projected = self.pre_project(src) #(N,L,64)
        input_padded = torch.zeros((src_projected.shape[0],
                                    src_projected.shape[1] + 2,
                                    src_projected.shape[2] + font.shape[2] + 2), dtype=torch.float).to(src.device)
        input_padded[:, src_projected.shape[1], src_projected.shape[2] + font.shape[2]] = 1.0  # add <sot> vector
        input_padded[:, src_projected.shape[1] + 1, src_projected.shape[2] + font.shape[2] + 1] = 1.0  # add <eot> vector
        input_padded[:, :src_projected.shape[1], :src_projected.shape[2]] = src_projected
        input_padded[:, :src_projected.shape[1], src_projected.shape[2]: src_projected.shape[2] + font.shape[2]] = font
        memory = self.transformer_encoder(input_padded)

        # training by teacher forcing
        if tgt_ids is not None:
            batch_indices = torch.arange(tgt_ids.shape[0]).unsqueeze(1).expand(tgt_ids.shape[0], tgt_ids.shape[1])
            # temporarily append one zero vector to facilitate tensor extraction
            memory_padded = torch.zeros((memory.shape[0], memory.shape[1]+1, memory.shape[2]), dtype=torch.float).to(src.device)
            memory_padded[:, :-1, :] = memory
            tgt_tensors = memory_padded[batch_indices, tgt_ids]
            query_tensors = memory[torch.arange(src.shape[0]), query_id].unsqueeze(1).expand(-1, tgt_tensors.shape[1], -1)
            decoder_input_cat = torch.cat([tgt_tensors, query_tensors], dim=-1)
            decoder_input = self.reduce_decoder_input_dim(decoder_input_cat)
            # decoder_input = self.pos_decoder(decoder_input)

            tgt_key_padding_mask = self.generate_padding_mask(tgt_ids, tgt_len).to(src.device)
            # tgt_mask = self.generate_attention_mask(decoder_input.shape[1]).to(src.device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

            decoder_hidden = self.transformer_decoder(decoder_input,
                                                      memory,
                                                      # memory_key_padding_mask=src_padding_mask,
                                                      tgt_mask=tgt_mask,
                                                      tgt_key_padding_mask=tgt_key_padding_mask)

            # decoder_output = self.output_layer(decoder_hidden)

            log_pointer_scores = self.pointer(decoder_hidden, memory)

        # inference or training without teacher forcing
        else:
            query_tensors = memory[torch.arange(memory.shape[0]), query_id].unsqueeze(1)

            decoder_input_cat = torch.cat([memory[:, 16, :].unsqueeze(1), query_tensors], dim=-1)  # <sot; query>

            decoder_input = self.reduce_decoder_input_dim(decoder_input_cat)
            # decoder_input = self.pos_decoder(decoder_input)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

            log_pointer_scores = torch.zeros((memory.shape[0], memory.shape[1], memory.shape[1]), dtype=torch.float).to(src.device)

            for t in range(memory.shape[1]-1):

                decoder_hidden = self.transformer_decoder(decoder_input,
                                                          memory,
                                                          # memory_key_padding_mask=src_padding_mask,
                                                          tgt_mask=tgt_mask)
                log_pointer_score = self.pointer(decoder_hidden, memory)
                log_pointer_scores[:, t, :] = log_pointer_score[:, t, :]

                next_id = torch.argmax(log_pointer_score[:, t, :], dim=-1)
                next_input = memory[torch.arange(memory.shape[0]), next_id]
                if t == memory.shape[1]-2:
                    break
                else:
                    next_input_cat = torch.cat([next_input.unsqueeze(1), query_tensors], dim=-1)
                    decoder_input = torch.cat([decoder_input, self.reduce_decoder_input_dim(next_input_cat)], dim=1)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

        return log_pointer_scores


class GrouperTransformerEncoderDecoderLinearNoFont(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=64, nhead=2, num_encoder_layers=1, num_decoder_layers=1):
        super(GrouperTransformerEncoderDecoderLinearNoFont, self).__init__()

        self.type = 'GrouperTransformerEncoderDecoderLinearNoFont'
        self.pre_project = nn.Linear(feature_dim, hidden_dim)  # shared across encoder and decoder
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.pos_decoder = PositionalEncoding(hidden_dim*2)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                        dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.post_project = nn.Linear(hidden_dim, hidden_dim*2)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim*2, nhead=nhead,
                                                        dim_feedforward=hidden_dim*4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.special_token_embedding = nn.Embedding(3, hidden_dim, padding_idx=2)  # 0: <sot>, 1: <eot>, 2: <pad>
        self.output_layer = nn.Linear(hidden_dim*2, 18)
        print('{} loaded.'.format(self.type))

    def generate_attention_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        # mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

    def generate_padding_mask(self, tgt_ids, tgt_len):
        tgt_key_padding_mask = torch.ones_like(tgt_ids, dtype=torch.bool)
        for i in range(tgt_len.shape[0]):
            tgt_key_padding_mask[i, :tgt_len[i] + 2] = False
        return tgt_key_padding_mask

    def forward(self, src, query_id, tgt_ids=None, tgt_len=None):

        # src: (N,L,f_sz)
        src = self.pre_project(src)
        sot_emb = self.special_token_embedding(torch.tensor([0]).to(src.device)).expand(src.shape[0], -1, -1)
        eot_emb = self.special_token_embedding(torch.tensor([1]).to(src.device)).expand(src.shape[0], -1, -1)
        pad_emb = self.special_token_embedding(torch.tensor([2]).to(src.device)).expand(src.shape[0], -1, -1)
        src = torch.cat([src, sot_emb, eot_emb], dim=1)
        src_padded = torch.cat([src, pad_emb], dim=1)  # padding to create a dictionary
        src_pos = self.pos_encoder(src)
        memory = self.transformer_encoder(src_pos)
        memory = self.post_project(memory)

        # src_padded: (N,L+3,h_dim)  src,memory: (N,L+2,h_dim)  query_id: (N,1)  tgt_ids: (N,L+2)
        # training by teacher forcing
        if tgt_ids is not None:
            batch_indices = torch.arange(tgt_ids.shape[0]).unsqueeze(1).expand(tgt_ids.shape[0], tgt_ids.shape[1])
            tgt_tensors = src_padded[batch_indices, tgt_ids]
            query_tensors = src[torch.arange(src.shape[0]), query_id].unsqueeze(1).expand(-1, tgt_tensors.shape[1], -1)
            decoder_input = torch.cat([tgt_tensors, query_tensors], dim=-1)
            decoder_input = self.pos_decoder(decoder_input)

            tgt_key_padding_mask = self.generate_padding_mask(tgt_ids, tgt_len).to(src.device)
            tgt_mask = self.generate_attention_mask(decoder_input.shape[1]).to(src.device)

            decoder_hidden = self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

            decoder_output = torch.nn.functional.log_softmax(self.output_layer(decoder_hidden), dim=-1)
        # inference or training without teacher forcing
        else:
            # initialize decoder input (position encodings, query tensors)
            decoder_input = torch.zeros((src.shape[0], src.shape[1]-1, src.shape[2]*2), dtype=torch.float).to(src.device)

            query_tensors = src[torch.arange(src.shape[0]), query_id].unsqueeze(1)
            decoder_input[:, :, src.shape[2]:] += query_tensors.expand(-1, decoder_input.shape[1], -1)
            decoder_input[:, 0, :src.shape[2]] += sot_emb.squeeze(1)
            decoder_input = self.pos_decoder(decoder_input)

            tgt_mask = self.generate_attention_mask(decoder_input.shape[1]).to(src.device)

            decoder_output = torch.zeros((src.shape[0], src.shape[1]-1, self.output_layer.out_features), dtype=torch.float).to(src.device)

            for t in range(src.shape[1]-1):

                decoder_hidden = self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)
                decoder_output[:, t] = torch.nn.functional.log_softmax(self.output_layer(decoder_hidden)[:, t], dim=-1)

                next_id = torch.argmax(decoder_output[:,t], dim=-1)
                next_input = src_padded[torch.arange(src.shape[0]), next_id]
                if t == src.shape[1]-2:
                    break
                else:
                    decoder_input[:, t+1, :src.shape[2]] += next_input

        return decoder_output

class GrouperTransformerEncoderWithFont(nn.Module):
    def __init__(self, feature_size=16, dim_hidden=128, num_heads=2, dropout=0.1):
        super(GrouperTransformerEncoderWithFont, self).__init__()

        self.type = 'GrouperTransformerEncoderWithFont'
        self.projection = nn.Linear(feature_size * 2, dim_hidden*2)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim_hidden*4, num_heads=num_heads, batch_first=True)

        self.layer_norm = nn.LayerNorm(dim_hidden*4)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim_hidden*4, dim_hidden*4 * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden*4 * 2, dim_hidden*4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(dim_hidden*4, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )

        print('{} loaded.'.format(self.type))

    def forward(self, queries, neighbors, queries_font, neighbors_font):

        # queries: (N,1,feature_sz)  sequences: (N,L,feature_sz)  queries_font: (N,1,font_emd_dim)  sequences_font: (N,L,font_emb_dim)
        queries_expanded = queries.expand(-1, neighbors.size(1), -1)
        input_concat = torch.cat([queries_expanded, neighbors], dim=-1)
        input_concat = self.projection(input_concat)

        queries_font_expanded = queries_font.expand(-1, neighbors_font.size(1), -1)
        input_concat = torch.cat([input_concat, queries_font_expanded], dim=-1)
        input_concat = torch.cat([input_concat, neighbors_font], dim=-1)

        attn_output, _ = self.multihead_attn(input_concat, input_concat, input_concat)
        attn_output = input_concat + self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output)
        attn_output = attn_output + self.feed_forward(attn_output)
        attn_output = self.layer_norm(attn_output)

        output = self.classifier(attn_output)  # (N,L,1)

        return output


class GrouperTransformerEncoderNoFont(nn.Module):
    def __init__(self, feature_size=16, dim_hidden=64, num_heads=2, dropout=0.1):
        super(GrouperTransformerEncoderNoFont, self).__init__()

        self.type = 'GrouperTransformerEncoderNoFont'
        self.projection = nn.Linear(feature_size * 2, dim_hidden)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)

        self.layer_norm = nn.LayerNorm(dim_hidden)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden * 2, dim_hidden)
        )

        self.classifier = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(),
            nn.Linear(dim_hidden // 2, 1),
            nn.Sigmoid()
        )

        print('{} loaded'.format(self.type))

    def forward(self, queries, neighbors):

        # queries: (N,1,feature_sz)  sequences: (N,L,feature_sz)
        queries_expanded = queries.expand(-1, neighbors.size(1), -1)
        input_concat = torch.cat([queries_expanded, neighbors], dim=-1)
        input_concat = self.projection(input_concat)

        attn_output, _ = self.multihead_attn(input_concat, input_concat, input_concat)
        attn_output = input_concat + self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output)
        attn_output = attn_output + self.feed_forward(attn_output)
        attn_output = self.layer_norm(attn_output)

        output = self.classifier(attn_output)  # (N,L,1)

        return output


class GrouperTransformerEncoderOnlyFont(nn.Module):
    def __init__(self, feature_size=16, dim_hidden=128, num_heads=2, dropout=0.1):
        super(GrouperTransformerEncoderOnlyFont, self).__init__()

        self.type = 'GrouperTransformerEncoderOnlyFont'

        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim_hidden*2, num_heads=num_heads, batch_first=True)

        self.layer_norm = nn.LayerNorm(dim_hidden*2)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim_hidden*2, dim_hidden*2 * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden*2 * 2, dim_hidden*2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(dim_hidden*2, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )

        print('{} loaded.'.format(self.type))

    def forward(self, queries_font, neighbors_font):

        # queries_font: (N,1,font_emd_dim)  sequences_font: (N,L,font_emb_dim)
        queries_font_expanded = queries_font.expand(-1, neighbors_font.size(1), -1)
        input_concat = torch.cat([neighbors_font, queries_font_expanded], dim=-1)

        attn_output, _ = self.multihead_attn(input_concat, input_concat, input_concat)
        attn_output = input_concat + self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output)
        attn_output = attn_output + self.feed_forward(attn_output)
        attn_output = self.layer_norm(attn_output)

        output = self.classifier(attn_output)  # (N,L,1)

        return output


class DatasetClass(Dataset):
    def __init__(self, data):

        self.data = data
        self.sot_id = 16
        self.eot_id = 17
        self.pad_id = 18

        self.max_source_length = 16
        self.max_source_length_include_spec = self.max_source_length + 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return (torch.tensor(self.data[idx]['query_text'], dtype=torch.long),
                torch.tensor(self.data[idx]['query_bezier_centralized'], dtype=torch.float),
                torch.tensor(self.data[idx]['neighbour_text'], dtype=torch.long),
                torch.tensor(self.data[idx]['neighbour_bezier_centralized'], dtype=torch.float),
                torch.tensor(self.data[idx]['neighbour_of_the_same_group'], dtype=torch.float),
                torch.tensor(self.data[idx]['query_bezier'], dtype=torch.float),
                torch.tensor(self.data[idx]['neighbour_bezier'], dtype=torch.float),
                torch.tensor(self.data[idx]['query_font'], dtype=torch.float),
                torch.tensor(self.data[idx]['neighbour_font'], dtype=torch.float),
                torch.tensor(self.data[idx]['source_text'], dtype=torch.long),
                torch.tensor(self.data[idx]['source_bezier_centralized'], dtype=torch.float),
                torch.tensor(self.data[idx]['source_bezier'], dtype=torch.float),
                torch.tensor(self.data[idx]['source_toponym_mask'], dtype=torch.long),
                torch.tensor(self.data[idx]['source_font'], dtype=torch.float),
                torch.tensor(self.data[idx]['toponym_id_sorted_in_source'], dtype=torch.long),
                torch.tensor(self.data[idx]['query_id_in_source'], dtype=torch.long),
                torch.tensor(self.data[idx]['img_id'], dtype=torch.long),
                torch.tensor(self.data[idx]['toponym_len'], dtype=torch.long))



def trainer(train_set, test_set):

    train_set = DatasetClass(train_set)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

    test_set = DatasetClass(test_set)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Create the model instance
    model = GrouperTransformerEncoderNoFont()

    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(model, train_dataloader, test_dataloader, optimizer, criterion, 1, device)

    return model, test_dataloader


def train(model, train_dataloader, test_dataloader, optimizer, criterion, epochs, device):

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(tqdm(train_dataloader)):

            batch = tuple(tensor.to(device) for tensor in batch)

            (query_text,
             query_bezier_centralized,
             neighbour_text,
             neighbour_bezier_centralized,
             neighbour_of_the_same_group,
             query_bezier,
             neighbour_bezier,
             query_font,
             neighbour_font,
             source_text,
             source_bezier_centralized,
             source_bezier,
             source_toponym_mask,
             source_font,
             toponym_id_sorted_in_source,
             query_id_in_source,
             img_id,
             toponym_len) = batch

            optimizer.zero_grad()

            if model.type == 'GrouperTransformerEncoderNoFont':
                outputs = model(query_bezier_centralized.unsqueeze(1), neighbour_bezier_centralized)
                loss = criterion(outputs, neighbour_of_the_same_group.unsqueeze(-1))
            elif model.type == 'GrouperTransformerEncoderWithFont':
                outputs = model(query_bezier_centralized.unsqueeze(1), neighbour_bezier_centralized, query_font.unsqueeze(1), neighbour_font)
                loss = criterion(outputs, neighbour_of_the_same_group.unsqueeze(-1))
            elif model.type == 'GrouperTransformerEncoderOnlyFont':
                outputs = model(query_font.unsqueeze(1), neighbour_font)
                loss = criterion(outputs, neighbour_of_the_same_group.unsqueeze(-1))
            elif model.type == 'GrouperTransformerEncoderDecoderAttentionNoFont':
                # outputs = model(source_bezier_centralized, query_id_in_source)
                outputs = model(source_bezier_centralized, query_id_in_source, toponym_id_sorted_in_source, toponym_len)
                loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), toponym_id_sorted_in_source[:, 1:].reshape(-1))
            elif model.type == 'GrouperTransformerEncoderDecoderAttentionWithFont':
                # outputs = model(source_bezier_centralized, query_id_in_source)
                outputs = model(source_bezier_centralized, source_font, query_id_in_source, toponym_id_sorted_in_source, toponym_len)
                loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), toponym_id_sorted_in_source[:, 1:].reshape(-1))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Log loss to TensorBoard
            if i % 10 == 9:    # log every 10 mini-batches
                step = epoch * len(train_dataloader) + i
                writer.add_scalar('Training loss', running_loss / 10, step)
                print('Training loss: {}'.format(running_loss / 10))
                running_loss = 0.0

        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}')

        evaluate(model, test_dataloader, criterion, device, step=epoch * len(train_dataloader), writer=writer)

        torch.save(model, './checkpoints/grouper_model_v1_epoch{}.pth'.format(epoch + 1))

    writer.close()


def evaluate(model, dataloader, criterion, device, step=None, writer=None, plot=None):

    model.eval()
    eval_loss = 0.0
    predictions_all = []
    labels_all = []
    num_no_eot = 0
    num_repetition = 0
    with torch.no_grad():

        for i, batch in enumerate(tqdm(dataloader)):

            batch = tuple(tensor.to(device) for tensor in batch)

            (query_text,
             query_bezier_centralized,
             neighbour_text,
             neighbour_bezier_centralized,
             neighbour_of_the_same_group,
             query_bezier,
             neighbour_bezier,
             query_font,
             neighbour_font,
             source_text,
             source_bezier_centralized,
             source_bezier,
             source_toponym_mask,
             source_font,
             toponym_id_sorted_in_source,
             query_id_in_source,
             img_id,
             toponym_len) = batch

            if model.type == 'GrouperTransformerEncoderNoFont':
                outputs = model(query_bezier_centralized.unsqueeze(1), neighbour_bezier_centralized)
                loss = criterion(outputs, neighbour_of_the_same_group.unsqueeze(-1))
                predictions = (outputs.cpu().flatten() >= 0.5).int().tolist()
                labels_flattened = neighbour_of_the_same_group.cpu().flatten().int().tolist()
            elif model.type == 'GrouperTransformerEncoderWithFont':
                outputs = model(query_bezier_centralized.unsqueeze(1), neighbour_bezier_centralized,
                                query_font.unsqueeze(1), neighbour_font)
                loss = criterion(outputs, neighbour_of_the_same_group.unsqueeze(-1))
                predictions = (outputs.cpu().flatten() >= 0.5).int().tolist()
                labels_flattened = neighbour_of_the_same_group.cpu().flatten().int().tolist()
            elif model.type == 'GrouperTransformerEncoderOnlyFont':
                outputs = model(query_font.unsqueeze(1), neighbour_font)
                loss = criterion(outputs, neighbour_of_the_same_group.unsqueeze(-1))
                predictions = (outputs.cpu().flatten() >= 0.5).int().tolist()
                labels_flattened = neighbour_of_the_same_group.cpu().flatten().int().tolist()
            elif 'GrouperTransformerEncoderDecoder' in model.type:
                if model.type == 'GrouperTransformerEncoderDecoderAttentionNoFont':
                    outputs = model(source_bezier_centralized, query_id_in_source)
                elif model.type == 'GrouperTransformerEncoderDecoderAttentionWithFont':
                    outputs = model(source_bezier_centralized, source_font, query_id_in_source)

                loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]),
                                 toponym_id_sorted_in_source[:, 1:].reshape(-1))

                output_id_in_source = torch.argmax(outputs, dim=-1)

                try:
                    eot_id = output_id_in_source.flatten().tolist().index(17)
                except:
                    eot_id = len(output_id_in_source.flatten().tolist())
                    num_no_eot += 1

                predicted_word_id_in_source = output_id_in_source.flatten().cpu().tolist()[:eot_id]

                if len(predicted_word_id_in_source) != len(set(predicted_word_id_in_source)):
                    num_repetition += 1

                predictions = []
                predicted_neighbour_order = []
                for i_ in range(16):
                    if i_ != query_id_in_source.cpu().item():
                        if i_ in predicted_word_id_in_source:
                            predictions.append(1)
                            predicted_neighbour_order.append(predicted_word_id_in_source.index(i_))
                        else:
                            predictions.append(0)
                            predicted_neighbour_order.append(-1)

                try:
                    predicted_query_order = predicted_word_id_in_source.index(query_id_in_source.cpu().item())
                except:
                    predicted_query_order = -1

                labels_flattened = [tm for i, tm in enumerate(source_toponym_mask.squeeze(0).cpu().int().tolist()) if i != query_id_in_source.cpu().item()]


            if plot is not None:
                if 'GrouperTransformerEncoderDecoder' in model.type:
                    grouper.predict_plot(query_pts=query_bezier.squeeze(0).squeeze(0).cpu().tolist(),
                                        query_text=decode_text_96(query_text.squeeze(0).tolist()),
                                        neighbour_pts=[sb for i, sb in enumerate(source_bezier.squeeze(0).cpu().tolist()) if i != query_id_in_source.cpu().item()],
                                        neighbour_text=[decode_text_96(text) for i, text in enumerate(source_text.squeeze(0).tolist()) if i != query_id_in_source.cpu().item()],
                                        gt_label=labels_flattened,
                                        predicted_label=predictions,
                                        img_id=img_id.cpu().item(),
                                        img_name=i,
                                        predicted_query_order=predicted_query_order,
                                        predicted_neighbour_order=predicted_neighbour_order)
                else:
                    grouper.predict_plot(query_pts=query_bezier.squeeze(0).squeeze(0).cpu().tolist(),
                                        query_text=decode_text_96(query_text.squeeze(0).tolist()),
                                        neighbour_pts=neighbour_bezier.squeeze(0).cpu().tolist(),
                                        neighbour_text=[decode_text_96(text) for text in neighbour_text.squeeze(0).tolist()],
                                        gt_label=labels_flattened,
                                        predicted_label=predictions,
                                        img_id=img_id.cpu().item(),
                                        img_name=i)

            eval_loss += loss.item()

            predictions_all.extend(predictions)
            labels_all.extend(labels_flattened)

    report = classification_report(labels_all, predictions_all)
    print("Classification Report:\n", report)
    print("ROC/AUC:\n", roc_auc_score(labels_all, predictions_all))
    print(f'Test Loss: {eval_loss/len(dataloader):.4f}')
    print('Num of test samples: {}'.format(len(dataloader)))
    print('Num of no eot: {}'.format(num_no_eot))
    print('Num of repitition: {}'.format(num_repetition))

    if writer and step:
        writer.add_scalar('Test loss', eval_loss / len(dataloader), step)


if __name__ == '__main__':

    grouper = GrouperDataset("train_images")
    grouper.load_annotations_from_file("train_96voc_embed_24.json")

    train_set, test_set = grouper.get_train_test_set(train_ratio=0.9, sample_ratio=1.0, random_seed=42)
    train_set = DatasetClass(train_set)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_set = DatasetClass(test_set)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    # model = GrouperTransformerEncoderNoFont()
    # model = GrouperTransformerEncoderDecoderLinearNoFont()
    # model = GrouperTransformerEncoderDecoderAttentionNoFont()
    model = GrouperTransformerEncoderDecoderAttentionWithFont()
    # model = GrouperTransformerEncoderWithFont()
    # model = GrouperTransformerEncoderOnlyFont()
    if 'GrouperTransformerEncoderDecoder' in model.type:
        criterion = nn.NLLLoss(ignore_index=18)
    else:
        criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    model.to(device)
    print('Device: {}'.format(device))

    train(model, train_dataloader, test_dataloader, optimizer, criterion, 5, device)

    evaluate(model, test_dataloader, criterion, device, step=None, writer=None, plot=grouper)




    # grouper = GrouperDataset("train_images")
    # grouper.load_annotations_from_file("train_96voc_embed.json")
    #
    # train_set, test_set = grouper.get_train_test_set(train_ratio=0.9, sample_ratio=1, random_seed=42)
    # train_set = DatasetClass(train_set)
    # train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    # test_set = DatasetClass(test_set)
    # test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    #
    # # model = GrouperTransformerEncoderDecoderAttentionNoFontPadded()
    # model = torch.load('./checkpoints/grouper_model_v1_epoch2.pth')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # print('Device: {}'.format(device))
    # if 'GrouperTransformerEncoderDecoder' in model.type:
    #     criterion = nn.NLLLoss(ignore_index=18)
    # else:
    #     criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # evaluate(model, test_dataloader, criterion, device, step=None, writer=None, plot=None)