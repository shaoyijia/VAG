import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


# distillation ########################
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


# contrastive ########################
# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class MyContrastive(nn.Module):
    # https://github.com/facebookresearch/moco/blob/3631be074a0a14ab85c206631729fe035e54b525/moco/builder.py#L155
    def __init__(self):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MyContrastive, self).__init__()
        self.n_gpu = torch.cuda.device_count()
        self.T = 1
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.sup_con = SupConLoss()

    def forward(self, x, order_x=None, labels=None, con_type=None):
        """
        labels indicate sample label
        x include all samples to be contrast
        """
        if self.n_gpu > 1:
            x = torch.cat(GatherLayer.apply(x), dim=0)
            if order_x is not None:
                order_x = torch.cat(GatherLayer.apply(order_x), dim=0)
            elif labels is not None:
                labels = torch.cat(GatherLayer.apply(labels), dim=0)

        if con_type == 'supervised':
            loss = self.sup_con(x, labels)
        elif con_type == 'unsupervised':
            loss = self.sup_con(x)
        elif con_type == 'soft_contrast':
            loss = self.unsupervised_loss(x, order_x, labels)

        return loss

    # Let's use forward to enable distributed training
    def unsupervised_loss(self, aug_x, order_x, output_label):
        logits = torch.einsum('nci,nkc->nki', [aug_x.unsqueeze(-1), order_x.permute(0, 2, 1)]).squeeze(
            -1)  # -dim=1 as number of order_x

        # apply temperature
        logits /= self.T

        contrast_loss = self.bce(logits, output_label)

        return contrast_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
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
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1 = D1.view(D1.size(0), -1)
        D1_norm = torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm = D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2 = D2.view(D2.size(0), -1)
        D2_norm = torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm = D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
