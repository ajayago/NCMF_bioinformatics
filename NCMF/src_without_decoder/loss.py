import torch
import math
import torch.nn as nn
import numpy as np


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class ZINBLoss(nn.Module):
    def __init__(self, reduction='none', eps=1e-10):
        super().__init__()
        self.eps = eps
        self.scale_factor = 1.0
        self.reduction = reduction

    def forward(self, y_pred, theta, pi, y_true, lamda, mask=None):
        if mask is not None:
            y_true = torch.masked_select(y_true, mask)
            y_pred = torch.masked_select(y_pred, mask)
            theta = torch.masked_select(theta, mask)
            pi = torch.masked_select(pi, mask)

        nb_case = self.nb_case_loss(y_true, y_pred, theta)
        zero_case = self.zero_case_loss(y_true, y_pred, theta, pi)

        t1 = torch.where(torch.less(y_true, 1e-8), zero_case, nb_case)
        t2 = lamda * torch.square(pi)
        loss = t1 + t2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def nb_case_loss(self, y_true, y_pred, theta):
        theta = torch.clamp(theta, max=1e6)
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true +
                                                           1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + \
            (y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        nb_case = t1 + t2
        return nb_case

    def zero_case_loss(self, y_true, y_pred, theta, pi):
        zero_nb = torch.pow(theta / (theta + y_pred + self.eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + self.eps)
        return zero_case

class ZINORMLoss(nn.Module):
    def __init__(self, reduction='none', eps=1e-10):
        super().__init__()
        self.eps = eps
        self.scale_factor = 1.0
        self.reduction = reduction
        self.constant_pi = torch.acos(torch.zeros(1)).item() * 2


    def forward(self, y_pred, theta, pi, y_true, lamda, mask=None):
        if mask is not None:
            y_true = torch.masked_select(y_true, mask)
            y_pred = torch.masked_select(y_pred, mask)
            theta = torch.masked_select(theta, mask)
            pi = torch.masked_select(pi, mask)

        zero_case = self.zero_case_loss(y_pred, theta, pi, y_true)
        norm_case = self.norm_case_loss(y_pred, theta, pi, y_true)

        t1 = torch.where(torch.less(y_true, 1e-8), zero_case, norm_case)
        t2 = lamda * torch.square(pi)
        loss = t1 + t2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def norm_case_loss(self, y_pred, theta, pi, y_true):
        theta = torch.clamp(theta, max=1e6)
        t1 = -torch.log(1.0 - pi)
        t2 = -0.5 * torch.log(2.0 * self.constant_pi * theta) - \
            torch.square(y_true - y_pred) / ((2 * theta) + self.eps)
        norm_case = t1 - t2
        return norm_case

    def zero_case_loss(self, y_pred, theta, pi, y_true):
        zero_norm = 1.0 / torch.sqrt(2.0 * pi * theta + self.eps) * torch.exp(-0.5 * (
            (0. - y_pred) ** 2) / theta + self.eps)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_norm) + self.eps)
        return zero_case

class KLDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD


class RMSELoss(nn.Module):
    def __init__(self, reduction='none', eps=1e-10):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_hat, y, mask=None):
        if mask is not None:
            y_hat = torch.masked_select(y_hat, mask)
            y = torch.masked_select(y, mask)
        loss = torch.sqrt(self.mse(y_hat, y) + self.eps)

        if self.reduction == 'mean':
            return loss.mean()
        return loss

def make_train_loss_function_ZINB():
    rec_loss_func = ZINBLoss(reduction='sum')
    kld_loss_func = KLDLoss()
    ncf_loss_func = ZINBLoss(reduction='mean')

    def vae_loss_func(y_pred, theta, pi, y_true, lamda, mask, mu, logvar, beta):
        rec = rec_loss_func(y_pred, theta, pi, y_true, lamda, mask)
        kld = kld_loss_func(mu, logvar)
        n = mask.sum()
        loss = 1 / n * (rec + beta * kld)
        return loss

    def loss_function(row_entities, col_entities, XP_block, Xs, XTs, X_block, lamda, X_masks, XT_masks, X_block_mask, beta=1):
        row_losses = loss_sum(vae_loss_func, row_entities,
                              Xs, lamda, X_masks, beta)
        col_losses = loss_sum(vae_loss_func, col_entities,
                              XTs, lamda, XT_masks, beta)
        rec_loss = ncf_loss_func(
            XP_block['M_bar'], XP_block['Theta'], XP_block['Pi'],
            X_block, lamda,
            X_block_mask
        )
        return row_losses, col_losses, rec_loss
    return loss_function

def make_train_loss_function_ZINORM():
    rec_loss_func = ZINORMLoss(reduction='sum')
    kld_loss_func = KLDLoss()
    ncf_loss_func = ZINORMLoss(reduction='mean')

    def vae_loss_func(y_pred, theta, pi, y_true, lamda, mask, mu, logvar, beta):
        rec = rec_loss_func(y_pred, theta, pi, y_true, lamda, mask)
        kld = kld_loss_func(mu, logvar)
        n = mask.sum()
        loss = 1 / n * (rec + beta * kld)
        return loss

    def loss_function(row_entities, col_entities, XP_block, Xs, XTs, X_block, lamda, X_masks, XT_masks, X_block_mask, beta=1):
        row_losses = loss_sum(vae_loss_func, row_entities,
                              Xs, lamda, X_masks, beta)
        col_losses = loss_sum(vae_loss_func, col_entities,
                              XTs, lamda, XT_masks, beta)
        rec_loss = ncf_loss_func(
            XP_block['M_bar'], XP_block['Theta'], XP_block['Pi'],
            X_block, lamda,
            X_block_mask
        )
        return row_losses, col_losses, rec_loss
    return loss_function


def loss_sum(loss_func, entities, truths, lamda, masks, beta):
    loss = 0.
    for entity, truth, mask in zip(entities, truths, masks):
        loss += loss_func(
            entity['M_bar'], entity['Theta'], entity['Pi'],
            truth, lamda, mask,
            entity['mu'], entity['logvar'],
            beta
        )
    return loss

def make_valid_loss_function_ZINB():
    rec_loss_func = ZINBLoss(reduction='mean')
    return rec_loss_func

def make_valid_loss_function_ZINORM():
    rec_loss_func = ZINORMLoss(reduction='mean')
    return rec_loss_func


def anneal_beta(M, R, T, t, anneal_type='linear'):
    tau = (t % math.ceil(T / M)) / (T/M)
    if anneal_type == 'linear':
        beta = R * tau if tau <= R else 1
    elif anneal_type == 'cosine':
        beta = (1 - np.cos((tau * np.pi / 2) / R)) if tau <= R else 1
    else:
        raise RuntimeError(
            f'anneal type {anneal_type} unavailable. Use one of linear, cosine.')
    return beta
