import torch.utils.data as data
import torch.nn.utils
import numpy as np

class SAKTDataset(torch.utils.data.Dataset):
    def __init__(self, df, n_skill, max_len=200):
        super(SAKTDataset, self).__init__()
        self.df = df
        self.n_skill = n_skill
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        qids = self.df[0][idx].split(",")
        correct = self.df[1][idx].split(",")

        if len(qids) > self.max_len:
            qids = qids[-self.max_len :]
            correct = correct[-self.max_len :]

        qids = np.array(list(map(int, qids)))
        correct = np.array(list(map(int, correct)))

        qa = qids + correct * self.n_skill

        q = np.ones(self.max_len, dtype=int) * self.n_skill
        qa2 = np.ones(self.max_len, dtype=int) * (self.n_skill * 2 + 1)
        correct2 = np.ones(self.max_len, dtype=int) * -1
        mask = np.zeros(self.max_len, dtype=int)

        q[: len(qids)] = qids
        qa2[: len(qa)] = qa
        correct2[: len(correct)] = correct
        mask[: len(qa)] = np.ones(len(qa), dtype=int)

        return (
            torch.cat(
                (torch.LongTensor([2 * self.n_skill]), torch.LongTensor(qa2[:-1]))
            ),
            torch.LongTensor(q),
            torch.LongTensor(correct2),
            torch.LongTensor(mask),
        )
