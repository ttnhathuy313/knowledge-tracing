# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional


class SAKTLoss(nn.Module):
    def __init__(self, reduce="mean"):
        super(SAKTLoss, self).__init__()
        self.reduce = reduce

    def forward(self, logits, targets, qid, mask, device="cpu"):

        mask = mask.gt(0).view(-1)
        targets = targets.view(-1)

        logits = torch.masked_select(logits.view(-1), mask)
        targets = torch.masked_select(targets, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction=self.reduce
        )
        return loss