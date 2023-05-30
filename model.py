# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)

def train_epoch(model, train_iterator, optim, criterion, device="cpu"):
    model.train()

    for i, (qa, qid, labels, mask) in enumerate(train_iterator):
        qa, qid, labels, mask = (
            qa.to(device),
            qid.to(device),
            labels.to(device),
            mask.to(device),
        )

        optim.zero_grad()
        logits, _ = model(qid, qa)
        loss = criterion(logits, labels, qid, mask, device=device)
        loss.backward()
        optim.step()
        
def future_mask(seq_length):
    mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype("bool")
    return torch.from_numpy(mask)


def eval_epoch(model, test_iterator, criterion, eval_func, device="cpu"):
    model.eval()

    eval_loss = []
    preds, binary_preds, targets = [], [], []
    for i, (qa, qid, labels, mask) in enumerate(test_iterator):
        qa, qid, labels, mask = (
            qa.to(device),
            qid.to(device),
            labels.to(device),
            mask.to(device),
        )

        with torch.no_grad():
            logits, _ = model(qid, qa)

        loss = criterion(logits, labels, qid, mask, device=device)
        eval_loss.append(loss.detach().item())

        mask = mask.eq(1)
        pred, binary_pred, target = eval_func(logits, qid, labels, mask)
        preds.append(pred)
        binary_preds.append(binary_pred)
        targets.append(target)

    preds = np.concatenate(preds)
    binary_preds = np.concatenate(binary_preds)
    targets = np.concatenate(targets)

    auc_value = roc_auc_score(targets, preds)
    accuracy = accuracy_score(targets, binary_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        targets, binary_preds
    )
    pos_rate = np.sum(targets) / float(len(targets))
    print(
        "auc={0}, accuracy={1}, precision={2}, recall={3}, fscore={4}, pos_rate={5}".format(
            auc_value, accuracy, precision, recall, f_score, pos_rate
        )
    )



class FFN(nn.Module):
    def __init__(self, state_size=200, dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size
        self.dropout = dropout
        self.lr1 = nn.Linear(self.state_size, self.state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(self.state_size, self.state_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

class SAKTModel(nn.Module):
    def __init__(
        self, n_skill, embed_dim, dropout, num_heads=4, max_len=64, device="cpu"
    ):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.q_embed_dim = embed_dim
        self.qa_embed_dim = embed_dim
        self.pos_embed_dim = embed_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.max_len = max_len
        self.device = device

        self.q_embedding = nn.Embedding(
            n_skill + 1, self.q_embed_dim, padding_idx=n_skill
        )
        self.qa_embedding = nn.Embedding(
            2 * n_skill + 2, self.qa_embed_dim, padding_idx=2 * n_skill + 1
        )
        self.pos_embedding = nn.Embedding(self.max_len, self.pos_embed_dim)

        self.multi_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout
        )

        self.key_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.query_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.ffn = FFN(self.embed_dim)
        self.pred = nn.Linear(self.embed_dim, 1, bias=True)

    def forward(self, q, qa):
        qa = self.qa_embedding(qa)
        pos_id = torch.arange(qa.size(1)).unsqueeze(0).to(self.device)
        pos_x = self.pos_embedding(pos_id)
        qa = qa + pos_x
        q = self.q_embedding(q)

        q = q.permute(1, 0, 2)
        qa = qa.permute(1, 0, 2)

        attention_mask = future_mask(q.size(0)).to(self.device)
        attention_out, _ = self.multi_attention(q, qa, qa, attn_mask=attention_mask)
        attention_out = self.layer_norm1(attention_out + q)
        attention_out = attention_out.permute(1, 0, 2)

        x = self.ffn(attention_out)
        x = self.dropout_layer(x)
        x = self.layer_norm2(x + attention_out)
        x = self.pred(x)

        return x.squeeze(-1), None