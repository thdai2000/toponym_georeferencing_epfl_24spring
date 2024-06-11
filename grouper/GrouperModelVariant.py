import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from GrouperDataVariant import GrouperDataset
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

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor, mask: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:

        # (B, Nd, Ne, C) <- (B, Ne, C)
        encoder_transform = self.w1(x_encoder).unsqueeze(1).expand(-1, x_decoder.shape[1], -1, -1)
        # (B, Nd, 1, C) <- (B, Nd, C)
        decoder_transform = self.w2(x_decoder).unsqueeze(2)
        # (B, Nd, Ne) <- (B, Nd, Ne, C), (B, Nd, 1, C)
        prod = self.v(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        # Apply mask: positions with True are not allowed to attend
        prod = prod.masked_fill(mask.unsqueeze(1).expand(-1, prod.shape[1], -1), float('-inf'))
        # (B, Nd, Ne) <- (B, Nd, Ne)
        log_score = torch.nn.functional.log_softmax(prod, dim=-1)
        return log_score


class GrouperTransformerEncoderDecoderAttentionNoFontPadded(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=64, nhead=2, num_encoder_layers=1, num_decoder_layers=1):
        super(GrouperTransformerEncoderDecoderAttentionNoFontPadded, self).__init__()

        self.type = 'GrouperTransformerEncoderDecoderAttentionNoFontPadded'
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
        self.special_token_id = {'<sot>': 0, '<eot>': 1, '<pad>': 18}
        # self.output_layer = nn.Linear(hidden_dim*2, 18)
        self.pointer = PointerNetwork(hidden_dim)

        print('{} loaded.'.format(self.type))

    def forward(self, src, query_id, src_len, tgt_ids=None, tgt_len=None):

        # src: (N,L,f_sz)  src: (N,L+3,f_sz+2)  src_len: (N, 1)
        # append special tokens features to input, <sot> and <eot> are binary vectors
        src_padded = torch.zeros((src.shape[0], src.shape[1] + 2, src.shape[2] + 2), dtype=torch.float).to(src.device)
        src_padded[:, 0, src.shape[2]] = 1.0  # add <sot> vector
        src_padded[:, 1, src.shape[2] + 1] = 1.0  # add <eot> vector
        src_padded[:, 2:, :src.shape[2]] = src

        src_padded_emb = self.pre_project(src_padded)
        # src_padded_emb = self.pos_encoder(src_padded_emb)

        src_padding_mask = torch.zeros((src_padded_emb.shape[0], src_padded_emb.shape[1]), dtype=torch.bool)
        for i in range(src_padding_mask.shape[0]):
            src_padding_mask[i, src_len[i] + 2:] = True
        src_padding_mask = src_padding_mask.to(src.device)

        memory = self.transformer_encoder(src_padded_emb,
                                          # src_key_padding_mask=src_padding_mask
                                          )
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

            tgt_key_padding_mask = torch.zeros_like(tgt_ids, dtype=torch.bool)
            for i in range(tgt_len.shape[0]):
                tgt_key_padding_mask[i, tgt_len[i] + 1:] = True
            tgt_key_padding_mask = tgt_key_padding_mask.to(src.device)
            # tgt_mask = self.generate_attention_mask(decoder_input.shape[1]).to(src.device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

            decoder_hidden = self.transformer_decoder(decoder_input,
                                                      memory,
                                                      tgt_mask=tgt_mask,
                                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                                      memory_key_padding_mask=src_padding_mask
                                                      )

            # decoder_output = self.output_layer(decoder_hidden)

            log_pointer_scores = self.pointer(decoder_hidden, memory, src_padding_mask)

        # inference or training without teacher forcing
        else:
            query_tensors = memory[torch.arange(memory.shape[0]), query_id].unsqueeze(1)

            decoder_input_cat = torch.cat([memory[:, 0, :].unsqueeze(1), query_tensors], dim=-1)  # <sot; query>

            decoder_input = self.reduce_decoder_input_dim(decoder_input_cat)
            # decoder_input = self.pos_decoder(decoder_input)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(src.device)

            log_pointer_scores = torch.zeros((memory.shape[0], memory.shape[1], memory.shape[1]), dtype=torch.float).to(src.device)

            for t in range(memory.shape[1]-1):

                decoder_hidden = self.transformer_decoder(decoder_input,
                                                          memory,
                                                          memory_key_padding_mask=src_padding_mask,
                                                          tgt_mask=tgt_mask)
                log_pointer_score = self.pointer(decoder_hidden, memory, src_padding_mask)
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


class DatasetClassTrainingVariantLength(Dataset):
    def __init__(self, data):

        self.data = data
        self.sot_id = 0
        self.eot_id = 1

        self.max_source_length = 16
        self.max_source_length_include_spec = self.max_source_length + 2

        self.pad_id = self.max_source_length_include_spec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        source_bezier_centralized_tensor = torch.tensor(self.data[idx]['source_bezier_centralized'], dtype=torch.float)
        source_bezier_tensor = torch.tensor(self.data[idx]['source_bezier'], dtype=torch.float)
        source_font_tensor = torch.tensor(self.data[idx]['source_font'], dtype=torch.float)
        source_text_tensor = torch.tensor(self.data[idx]['source_text'], dtype=torch.long)
        source_length = source_bezier_centralized_tensor.shape[0]

        if source_length < self.max_source_length:
            num_to_pad = self.max_source_length - source_length
            feature_size = source_bezier_centralized_tensor.shape[1]
            font_size = source_font_tensor.shape[1]
            text_size = source_text_tensor.shape[1]
            bezier_padding_tensor = torch.zeros((num_to_pad, feature_size), dtype=torch.float)
            font_padding_tensor = torch.zeros((num_to_pad, font_size), dtype=torch.float)
            text_padding_tensor = torch.zeros((num_to_pad, text_size), dtype=torch.long)

            source_bezier_centralized_tensor = torch.cat([source_bezier_centralized_tensor, bezier_padding_tensor], dim=0)
            source_bezier_tensor = torch.cat([source_bezier_tensor, bezier_padding_tensor], dim=0)
            source_font_tensor = torch.cat([source_font_tensor, font_padding_tensor], dim=0)
            source_text_tensor = torch.cat([source_text_tensor, text_padding_tensor], dim=0)

        source_toponym_mask_tensor = torch.tensor(self.data[idx]['source_toponym_mask'], dtype=torch.long)
        source_toponym_mask_tensor = torch.cat([torch.tensor([0, 0]),
                                                source_toponym_mask_tensor,
                                                torch.tensor([0]).repeat(
                                                    self.max_source_length - source_length
                                                )], dim=-1)
        toponym_id_sorted_in_source_tensor = torch.tensor(self.data[idx]['toponym_id_sorted_in_source'], dtype=torch.long)
        toponym_id_sorted_in_source_tensor = torch.cat([torch.tensor([self.sot_id]),
                                                        toponym_id_sorted_in_source_tensor + 2,
                                                        torch.tensor([self.eot_id]),
                                                        torch.tensor([self.pad_id]).repeat(
                                                            self.max_source_length - self.data[idx]['toponym_len'])],
                                                       dim=-1)

        assert (len(source_text_tensor) == len(source_bezier_tensor) == len(source_bezier_centralized_tensor) ==
                len(source_font_tensor) == self.max_source_length)
        assert len(source_toponym_mask_tensor) == len(toponym_id_sorted_in_source_tensor) == self.max_source_length_include_spec

        return (torch.tensor(self.data[idx]['query_text'], dtype=torch.long),
                torch.tensor(self.data[idx]['query_bezier_centralized'], dtype=torch.float),
                torch.tensor(self.data[idx]['query_bezier'], dtype=torch.float),
                torch.tensor(self.data[idx]['query_font'], dtype=torch.float),
                source_text_tensor,
                source_bezier_centralized_tensor,
                source_bezier_tensor,
                source_toponym_mask_tensor,
                source_font_tensor,
                toponym_id_sorted_in_source_tensor,
                torch.tensor(self.data[idx]['query_id_in_source'], dtype=torch.long) + 2,
                torch.tensor(self.data[idx]['img_id'], dtype=torch.long),
                torch.tensor(self.data[idx]['toponym_len'], dtype=torch.long),
                torch.tensor(self.data[idx]['source_len'], dtype=torch.long))


class DatasetClassInference(Dataset):
    def __init__(self, data):

        self.data = data
        self.sot_id = 0
        self.eot_id = 1

        self.max_source_length = 16
        self.max_source_length_include_spec = self.max_source_length + 2

        self.pad_id = self.max_source_length_include_spec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        source_bezier_centralized_tensor = torch.tensor(self.data[idx]['source_bezier_centralized'], dtype=torch.float)

        source_length = source_bezier_centralized_tensor.shape[0]

        if source_length < self.max_source_length:
            num_to_pad = self.max_source_length - source_length
            feature_size = source_bezier_centralized_tensor.shape[1]
            bezier_padding_tensor = torch.zeros((num_to_pad, feature_size), dtype=torch.float)

            source_bezier_centralized_tensor = torch.cat([source_bezier_centralized_tensor, bezier_padding_tensor], dim=0)

        return (source_bezier_centralized_tensor,
                torch.tensor(self.data[idx]['query_id_in_source'], dtype=torch.long) + 2,
                torch.tensor(self.data[idx]['source_len'], dtype=torch.long))


def train(model, train_dataloader, test_dataloader, optimizer, criterion, epochs, device):

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(tqdm(train_dataloader)):

            batch = tuple(tensor.to(device) for tensor in batch)

            (query_text,
             query_bezier_centralized,
             query_bezier,
             query_font,
             source_text,
             source_bezier_centralized,
             source_bezier,
             source_toponym_mask,
             source_font,
             toponym_id_sorted_in_source,
             query_id_in_source,
             img_id,
             toponym_len,
             source_len) = batch

            optimizer.zero_grad()

            if model.type == 'GrouperTransformerEncoderDecoderAttentionNoFontPadded':
                # outputs = model(source_bezier_centralized, query_id_in_source)
                outputs = model(source_bezier_centralized, query_id_in_source, source_len, toponym_id_sorted_in_source, toponym_len)
                loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), toponym_id_sorted_in_source[:, 1:].reshape(-1))
            elif model.type == 'GrouperTransformerEncoderDecoderAttentionWithFont':
                # outputs = model(source_bezier_centralized, query_id_in_source)
                outputs = model(source_bezier_centralized, source_font, query_id_in_source, source_len, toponym_id_sorted_in_source, toponym_len)
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

        torch.save(model, './checkpoints/grouper_model_epoch{}.pth'.format(epoch+1))

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
             query_bezier,
             query_font,
             source_text,
             source_bezier_centralized,
             source_bezier,
             source_toponym_mask,
             source_font,
             toponym_id_sorted_in_source,
             query_id_in_source,
             img_id,
             toponym_len,
             source_len) = batch


            if 'GrouperTransformerEncoderDecoder' in model.type:
                if model.type == 'GrouperTransformerEncoderDecoderAttentionNoFontPadded':
                    outputs = model(source_bezier_centralized, query_id_in_source, source_len)
                elif model.type == 'GrouperTransformerEncoderDecoderAttentionWithFont':
                    outputs = model(source_bezier_centralized, source_font, query_id_in_source)

                loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]),
                                 toponym_id_sorted_in_source[:, 1:].reshape(-1))

                output_id_in_source = torch.argmax(outputs, dim=-1)

                try:
                    eot_id = output_id_in_source.flatten().tolist().index(1)
                except:
                    eot_id = len(output_id_in_source.flatten().tolist())
                    num_no_eot += 1

                predicted_word_id_in_source = output_id_in_source.flatten().cpu().tolist()[:eot_id]

                if len(predicted_word_id_in_source) != len(set(predicted_word_id_in_source)):
                    num_repetition += 1

                # source len is the length of source without special tokens, it includes query and neighbours
                predictions = []
                predicted_neighbour_order = []
                source_len_no_spec = source_len.cpu().item()
                for i_ in range(source_len_no_spec + 2):
                    if i_ != query_id_in_source.cpu().item() and i_ > 1:
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

                labels_flattened = [tm for i, tm in enumerate(source_toponym_mask.squeeze(0).cpu().int().tolist()[:source_len_no_spec + 2]) if i != query_id_in_source.cpu().item() and i > 1]

            if plot is not None:
                if 'GrouperTransformerEncoderDecoder' in model.type:
                    grouper.predict_plot(query_pts=query_bezier.squeeze(0).squeeze(0).cpu().tolist(),
                                        query_text=decode_text_96(query_text.squeeze(0).tolist()),
                                        neighbour_pts=[sb for i, sb in enumerate(source_bezier.squeeze(0).cpu().tolist()[:source_len_no_spec]) if i != query_id_in_source.cpu().item()-2],
                                        neighbour_text=[decode_text_96(text) for i, text in enumerate(source_text.squeeze(0).tolist()[:source_len_no_spec]) if i != query_id_in_source.cpu().item()-2],
                                        gt_label=labels_flattened,
                                        predicted_label=predictions,
                                        img_id=img_id.cpu().item(),
                                        img_name=i,
                                        predicted_query_order=predicted_query_order,
                                        predicted_neighbour_order=predicted_neighbour_order)

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

    # grouper = GrouperDataset("train_images")
    # grouper.load_annotations_from_file("train_96voc_embed.json")
    #
    # train_set, test_set = grouper.get_train_test_set(train_ratio=0.9, sample_ratio=1, random_seed=42)
    # train_set = DatasetClassTrainingVariantLength(train_set)
    # train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    # test_set = DatasetClassTrainingVariantLength(test_set)
    # test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    #
    # # model = GrouperTransformerEncoderNoFont()
    # # model = GrouperTransformerEncoderDecoderLinearNoFont()
    # model = GrouperTransformerEncoderDecoderAttentionNoFontPadded()
    # # model = GrouperTransformerEncoderDecoderAttentionWithFont()
    # # model = GrouperTransformerEncoderWithFont()
    # # model = GrouperTransformerEncoderOnlyFont()
    # if 'GrouperTransformerEncoderDecoder' in model.type:
    #     criterion = nn.NLLLoss(ignore_index=18)
    # else:
    #     criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # device = 'cpu'
    # model.to(device)
    # print('Device: {}'.format(device))
    #
    # train(model, train_dataloader, test_dataloader, optimizer, criterion, 3, device)
    #
    # evaluate(model, test_dataloader, criterion, device, step=None, writer=None, plot=grouper)



    grouper = GrouperDataset("train_images")
    grouper.load_annotations_from_file("train_96voc_embed.json")

    train_set, test_set = grouper.get_train_test_set(train_ratio=0.9, sample_ratio=1, random_seed=42)
    train_set = DatasetClassTrainingVariantLength(train_set)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_set = DatasetClassTrainingVariantLength(test_set)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    # model = GrouperTransformerEncoderDecoderAttentionNoFontPadded()
    model = torch.load('./checkpoints/grouper_model_epoch3.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('Device: {}'.format(device))
    if 'GrouperTransformerEncoderDecoder' in model.type:
        criterion = nn.NLLLoss(ignore_index=18)
    else:
        criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    evaluate(model, test_dataloader, criterion, device, step=None, writer=None, plot=None)