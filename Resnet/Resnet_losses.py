import os
import time
import datetime
import numpy as np

import torch
import torch.nn as nn


def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    loss_batch = torch.sum(- targets_prob * student_likelihood, dim=-1)
    return loss_batch.mean()


class ResnetLoss(nn.Module):
    def __init__(self, pred_weight, soft_weight, dist_weight, Ar, teacher=None):
        super().__init__()
        self.pred_weight = pred_weight #0 in minivit        
        self.soft_weight = soft_weight #1 in minivit
        self.dist_weight = dist_weight
        self.teacher = teacher
        self.Ar = Ar
        self.criterion_soft = soft_cross_entropy
    def loss_terms(self):
        return ['pred_loss', 'dist_loss', 'soft_loss', 'total_loss']
    def forward(self, input, output, label, add_dist=True):
        pred_y, dist = output
        with torch.no_grad():
            if (self.soft_weight > 0 and self.teacher is not None):
                self.teacher.eval()
                teacher_output = self.teacher(input)
                soft_loss = self.criterion_soft(pred_y, teacher_output)
            else:
                soft_loss = torch.zeros(1).to(input.device)
        pred_loss = nn.CrossEntropyLoss(reduction='sum')(pred_y, label)
        if not add_dist:
            total_loss = pred_loss * self.pred_weight + soft_loss * self.soft_weight
        else:
            total_loss = pred_loss * self.pred_weight + dist * self.dist_weight + soft_loss * self.soft_weight

        return total_loss, {
            'pred_loss':  pred_loss.detach().item(),
            'dist_loss':  dist.detach().item(),
            "soft_loss":  soft_loss.detach().item(),
            'total_loss': total_loss.detach().item()
        }