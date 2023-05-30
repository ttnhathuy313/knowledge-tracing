import torch
from dataset import SAKTDataset
from model import SAKTModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

arg_parser = argparse.ArgumentParser(description="train deep IRT model")
arg_parser.add_argument("--learning_rate",
                        dest="learning_rate",
                        default=0.001,
                        type=float,
                        required=False)
arg_parser.add_argument("--batch_size",
                        dest="batch_size",
                        default=64,
                        type=int,
                        required=False)
arg_parser.add_argument("--num_skill",
                        dest="num_skill",
                        default=100,
                        type=int,
                        required=False)
arg_parser.add_argument("--embed_dim",
                        dest="embed_dim",
                        default=200,
                        type=int,
                        required=False)
arg_parser.add_argument("--dropout",
                        dest="dropout",
                        default=0.2,
                        type=float,
                        required=False)
arg_parser.add_argument("--num_heads",
                        dest="num_heads",
                        default=5,
                        type=int,
                        required=False)
arg_parser.add_argument("--epoch",
                        dest="epoch",
                        default=15,
                        type=int,
                        required=False)
arg_parser.add_argument("--num_worker",
                        dest="num_worker",
                        default=0,
                        type=int,
                        required=False)
args = arg_parser.parse_args()

#load the model sakt_all.pth
MODEL_PATH = "./saved/sakt_weights.pth"
model = SAKTModel(args.num_skill, args.embed_dim, args.dropout, args.num_heads, max_len=128)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

train_df = pd.read_csv("./data/assist2015_train.csv",
                           header=None,
                           sep='\t')
train = SAKTDataset(train_df, 100, max_len=128)
train_dataloader = DataLoader(train,
                            batch_size=args.batch_size,
                            num_workers=args.num_worker,
                            shuffle=True)
for i, (qa, qid, labels, mask) in enumerate(train_dataloader):
    with torch.no_grad():
        logits, _ = model(qid, qa)
    logits = torch.sigmoid(logits)
    embeddings = model.q_embedding(qid).detach().numpy()
    # print(logits[0].round())
    # print(labels[0])
    # print(qa.shape) 
    break

def query(prev_qid, prev_correct, cur_qid):
    prev_qid = np.array(prev_qid)
    prev_correct = np.array(prev_correct)
    qa = prev_qid + 100 * prev_correct
    padding_qa = np.ones(128, dtype=np.int8) * 201
    padding_qa[:len(qa)] = qa
    qid = prev_qid.copy().tolist()
    qid.append(cur_qid)
    qid = np.array(qid)
    padding_qid = np.ones(128, dtype=np.int8) * 100
    padding_qid[:len(qid)] = qid 
    padding_qa = torch.LongTensor(np.array([padding_qa]))
    padding_qid = torch.LongTensor(np.array([padding_qid]))
    with torch.no_grad():
        logits, _ = model(padding_qid, padding_qa)
    logits = torch.sigmoid(logits)
    prob = logits[0][len(prev_qid)].detach().numpy()
    return int(prob * 100)
