import os

import torch
import torch.nn as nn
from bert.modeling import BertModel


class RoBertModel(nn.Module):
    def __init__(self, bert_model_dir,pre_trained_model):
        super(RoBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_dir,
                                              state_dict=torch.load(pre_trained_model)["state_dict"])

    def batched_index_select(self, t, dim, inds):
        dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))

        out = t.gather(dim, dummy)

        return out

    def forward(self, bert_ids, bert_idx=None):
        bert_mask = bert_ids.ne(0).long()
        bert_out = self.bert(bert_ids, bert_mask, output_all_encoded_layers=False)
        if bert_idx is not None:
            bert_x_out = self.batched_index_select(bert_out, dim=1, inds=bert_idx)
            return bert_x_out

        else:
            return bert_out


class RobertaForSequenceClassification(nn.Module):
    def __init__(self,bert_model_dir,pre_trained_model, classifier_dropout=0.3):
        super(RobertaForSequenceClassification, self).__init__()

        self.bert = RoBertModel(bert_model_dir,pre_trained_model)

        self.lstm = nn.LSTM(input_size=768, hidden_size=512, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(1024, 1)

        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, input_ids, attention_mask=None, labels=None):

        sequence_output = self.bert(input_ids)

        sentence_embeddings = sequence_output[:, 1:-1, :]

        lstm_output, (lstm_final_state, _) = self.lstm(sentence_embeddings)

        context = self.self_attention(lstm_output, lstm_final_state)

        context = self.dropout(context)
        logits = self.linear(context)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(input=logits, target=labels)
            return loss

        return logits


    def self_attention(self, lstm_output, lstm_final_state):
        hidden = lstm_final_state.view(-1, 512 * 2, 1)
        attn_score = torch.bmm(lstm_output, hidden).squeeze(2)
        attn_dist = F.softmax(attn_score)
        context = torch.bmm(lstm_output.transpose(1, 2), attn_dist.unsqueeze(2)).squeeze(2)

        return context
