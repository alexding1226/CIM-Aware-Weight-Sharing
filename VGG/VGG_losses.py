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


def cal_relation_loss(student_attn_list, teacher_attn_list, Ar): #Ar : default = 1 in minivit
    layer_num = len(student_attn_list)
    relation_loss = 0.
    for student_att, teacher_att in zip(student_attn_list, teacher_attn_list):
        B, N, Cs = student_att[0].shape
        _, _, Ct = teacher_att[0].shape
        for i in range(3):
            for j in range(3):
                # (B, Ar, N, Cs // Ar) @ (B, Ar, Cs // Ar, N)
                # (B, Ar) + (N, N)
                matrix_i = student_att[i].view(B, N, Ar, Cs//Ar).transpose(1, 2) / (Cs/Ar)**0.5
                matrix_j = student_att[j].view(B, N, Ar, Cs//Ar).permute(0, 2, 3, 1)
                As_ij = (matrix_i @ matrix_j) 

                matrix_i = teacher_att[i].view(B, N, Ar, Ct//Ar).transpose(1, 2) / (Ct/Ar)**0.5
                matrix_j = teacher_att[j].view(B, N, Ar, Ct//Ar).permute(0, 2, 3, 1)
                At_ij = (matrix_i @ matrix_j)
                relation_loss += soft_cross_entropy(As_ij, At_ij)
    return relation_loss/(9. * layer_num)


def cal_hidden_relation_loss(student_hidden_list, teacher_hidden_list):
    layer_num = len(student_hidden_list)
    B, N, Cs = student_hidden_list[0].shape
    _, _, Ct = teacher_hidden_list[0].shape
    hidden_loss = 0.
    for student_hidden, teacher_hidden in zip(student_hidden_list, teacher_hidden_list):
        student_hidden = torch.nn.functional.normalize(student_hidden, dim=-1)
        teacher_hidden = torch.nn.functional.normalize(teacher_hidden, dim=-1)
        student_relation = student_hidden @ student_hidden.transpose(-1, -2)
        teacher_relation = teacher_hidden @ teacher_hidden.transpose(-1, -2)
        hidden_loss += torch.mean((student_relation - teacher_relation)**2) * 49 #Window size x Window size
    return hidden_loss/layer_num


class VGGLoss(nn.Module):
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
                soft_loss = 0
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


class Distlossqkv_hidden_dist(nn.Module):
    def __init__(self,  qkv_weight,pred_weight,soft_weight,hidden_weight,dist_weight,teacher, Ar):
        super().__init__()
        self.qkv_weight = qkv_weight #1 in minivit
        self.pred_weight = pred_weight #0 in minivit
        self.soft_weight = soft_weight #1 in minivit
        self.hidden_weight = hidden_weight #1 in minivit
        self.dist_weight = dist_weight
        self.teacher = teacher
        self.Ar = Ar
        self.criterion_soft = soft_cross_entropy
    def loss_terms(self):
        return ['qkv_loss', 'pred_loss', 'soft_loss','hidden_loss','dist_loss', 'total_loss']
    def forward(self, input,output,label,add_dist=True):
        pred_y, student_attn_tuple,hid_tuple,dist = output
        with torch.no_grad():
            self.teacher.eval()
            teacher_output,teacher_attn_tuple,teacher_hid_tuple = self.teacher(input)
        pred_loss = nn.CrossEntropyLoss(reduction='sum')(pred_y, label)
        #print(torch.equal(student_attn_tuple, teacher_attn_tuple))
        #print(torch.equal(pred_y, teacher_output),"output")
        qkv_loss = cal_relation_loss(student_attn_tuple, teacher_attn_tuple, self.Ar)
        soft_loss = self.criterion_soft(pred_y, teacher_output)
        hidden_loss = cal_hidden_relation_loss(hid_tuple, teacher_hid_tuple)
        if dist == 0 or not add_dist:
            total_loss = pred_loss * self.pred_weight + qkv_loss * self.qkv_weight + soft_loss * self.soft_weight + hidden_loss * self.hidden_weight
        else:
            total_loss = pred_loss * self.pred_weight + qkv_loss * self.qkv_weight + soft_loss * self.soft_weight + hidden_loss * self.hidden_weight + dist * self.dist_weight 

        return total_loss, {
            'qkv_loss': qkv_loss.detach().item(),
            'pred_loss': pred_loss.detach().item(),
            'soft_loss': soft_loss.detach().item(),
            "hidden_loss": hidden_loss.detach().item(),
            "dist_loss": dist.detach().item(),
            "total_loss": total_loss.detach().item()
            
        }

class Distlossqkv_hidden(nn.Module):
    def __init__(self,  qkv_weight,pred_weight,soft_weight,hidden_weight,teacher, Ar):
        super().__init__()
        self.qkv_weight = qkv_weight #1 in minivit
        self.pred_weight = pred_weight #0 in minivit
        self.soft_weight = soft_weight #1 in minivit
        self.hidden_weight = hidden_weight #1 in minivit
        self.teacher = teacher
        self.Ar = Ar
        self.criterion_soft = soft_cross_entropy
    def loss_terms(self):
        return ['qkv_loss', 'pred_loss', 'soft_loss', 'total_loss','hidden_loss']
    def forward(self, input,output,label):
        pred_y, student_attn_tuple,hid_tuple = output
        with torch.no_grad():
            self.teacher.eval()
            teacher_output,teacher_attn_tuple,teacher_hid_tuple = self.teacher(input)
        pred_loss = nn.CrossEntropyLoss(reduction='sum')(pred_y, label)
        #print(torch.equal(student_attn_tuple, teacher_attn_tuple))
        #print(torch.equal(pred_y, teacher_output),"output")
        qkv_loss = cal_relation_loss(student_attn_tuple, teacher_attn_tuple, self.Ar)
        soft_loss = self.criterion_soft(pred_y, teacher_output)
        hidden_loss = cal_hidden_relation_loss(hid_tuple, teacher_hid_tuple)
        total_loss = pred_loss * self.pred_weight + qkv_loss * self.qkv_weight + soft_loss * self.soft_weight + hidden_loss * self.hidden_weight

        return total_loss, {
            'qkv_loss': qkv_loss.detach().item(),
            'pred_loss': pred_loss.detach().item(),
            'soft_loss': soft_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
            "hidden_loss": hidden_loss.detach().item()
        }
    
class Distlossqkv(nn.Module):
    def __init__(self,  qkv_weight,pred_weight,soft_weight,teacher, Ar):
        super().__init__()
        self.qkv_weight = qkv_weight #1 in minivit
        self.pred_weight = pred_weight #0 in minivit
        self.soft_weight = soft_weight #1 in minivit
        self.teacher = teacher
        self.Ar = Ar
        self.criterion_soft = soft_cross_entropy
    def loss_terms(self):
        return ['qkv_loss', 'pred_loss', 'soft_loss', 'total_loss','hidden_loss']
    def forward(self, input,output,label):
        pred_y, student_attn_tuple= output
        with torch.no_grad():
            self.teacher.eval()
            teacher_output,teacher_attn_tuple = self.teacher(input)
        pred_loss = nn.CrossEntropyLoss(reduction='sum')(pred_y, label)
        #print(torch.equal(student_attn_tuple, teacher_attn_tuple))
        #print(torch.equal(pred_y, teacher_output),"output")
        qkv_loss = cal_relation_loss(student_attn_tuple, teacher_attn_tuple, self.Ar)
        soft_loss = self.criterion_soft(pred_y, teacher_output)
        total_loss = pred_loss * self.pred_weight + qkv_loss * self.qkv_weight + soft_loss * self.soft_weight

        return total_loss, {
            'qkv_loss': qkv_loss.detach().item(),
            'pred_loss': pred_loss.detach().item(),
            'soft_loss': soft_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
        }
    
class CEloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
    def loss_terms(self):
        return ['pred_loss', 'total_loss']
    def forward(self, input,output,label):
        if len(output) == 2:
            pred_y, _ = output
        elif len(output) == 3:
            pred_y, _, _ = output
        else:
            pred_y = output
        pred_loss = self.criterion(pred_y, label)
        total_loss = pred_loss
        return total_loss, {
            'pred_loss': pred_loss.detach().item(),
            "total_loss": total_loss.detach().item()
        }