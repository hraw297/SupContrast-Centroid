"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='cent',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, centroid_ft=None, centroid_lbl=None, alpha=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if self.contrast_mode in ['eq1', 'eq2', 'eq3']:
            if type(centroid_ft) == type(None) or type(centroid_lbl) == type(None):
                raise ValueError('In `contrast_mode` eq1, eq2 and eq3 `centroid_lbl`'
                    'and `centroid_ft` cannot be None')
            if self.contrast_mode == 'eq1' and not alpha:
                raise ValueError('`alpha` must have positive value between 0. and 1.'
                    'when using eq1 as  `contrast_mode`')

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
                
            if self.contrast_mode in ['eq1', 'eq2']:
                c_mask = torch.eq(labels, centroid_lbl).float().to(device)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode in ['eq1', 'eq2']:
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            contrast_feature = torch.cat((contrast_feature, centroid_ft))
        elif self.contrast_mode == 'eq3':
            contrast_feature = torch.cat((contrast_feature, centroid_ft))
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        if self.contrast_mode == 'eq3':
            lbl = labels.squeeze(1).repeat(1, contrast_count).squeeze(0)
            labels = torch.cat((lbl, centroid_lbl)).contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.repeat(anchor_count, contrast_count)
        if self.contrast_mode in ['eq1', 'eq2']:
            c_mask = c_mask.repeat(anchor_count, 1)
            mask = torch.cat((mask, c_mask), dim=1)

        # mask-out self-contrast cases
        if self.contrast_mode == 'eq3':
            mask_count = batch_size * anchor_count + centroid_lbl.shape[0]
        else:
            mask_count = batch_size * anchor_count

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask_count).view(-1, 1).to(device),
            0
        ).to(device)
        
        mask = mask * logits_mask

        # compute log_prob
        if self.contrast_mode == 'eq1':
            c_m_i = batch_size * anchor_count
            c_inds = torch.arange(c_m_i, c_m_i + c_mask.shape[1]).unsqueeze(0)\
                .repeat(c_m_i, 1).to(device)
            
            logits_mask = torch.scatter(logits_mask, 1, c_inds, 0)
            c_logits_mask = torch.scatter(torch.zeros_like(mask).to(device), 1, c_inds, 1).to(device)

            exp_logits = torch.exp(logits) * logits_mask
            c_exp_logits = torch.exp(logits) * c_logits_mask

            log_prob = logits
            log_prob[:, :c_m_i] -= torch.log(exp_logits.sum(1, keepdim=True))
            log_prob[:, c_m_i:] *= alpha
            log_prob[:, c_m_i:] -= torch.log(c_exp_logits.sum(1, keepdim=True))
        else:
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if self.contrast_mode == 'eq3':
            loss = loss.mean()
        else:
            loss = loss.view(anchor_count, batch_size).mean()


        return loss
