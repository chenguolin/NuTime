import copy

from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from src.models.networks import *
from src.utils.util import concat_all_gather, is_dist_avail_and_initialized


class SimCLRModel(nn.Module):
    """SimCLR model for time series self-supervised learning.

    Referred from https://github.com/google-research/simclr.
    """
    def __init__(self, backbone: nn.Module, num_layers=2, use_bn=True, non_linear=True, tau=0.1):
        super(SimCLRModel, self).__init__()

        self.tau = tau
        self.backbone = backbone
        mlp_dim = self.backbone[-1].fc.in_features
        out_dim = self.backbone[-1].fc.out_features  # default: 128

        # add mlp projection head
        layers = []
        for _ in range(num_layers-1):
            layers += [
                nn.Linear(mlp_dim, mlp_dim, bias=False),
                nn.BatchNorm1d(mlp_dim) if use_bn else nn.Identity(),
                nn.ReLU() if non_linear else nn.Identity(),
            ]
        layers += [
            nn.Linear(mlp_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim) if use_bn else nn.Identity(),
        ]
        self.backbone[-1].fc = nn.Sequential(*layers)

    def forward(self, x1, x2):
        B = x1.shape[0]
        x = torch.cat([x1, x2], dim=0)
        # forward backbone
        features = self.backbone(x)  # (2B, D)
        # pairwise cosine similarity
        features = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(features, features.t())  # (2B, 2B)
        # labels for positive/negative pairs
        pair_labels = torch.eye(B, dtype=torch.long, device=features.device).repeat(2, 2)  # (2B, 2B)
        # mask out diagonal elements (i.e., self-pairs)
        mask = ~torch.eye(2*B, dtype=torch.bool, device=features.device)
        sim_matrix = sim_matrix[mask].view(2*B, -1)  # (2B, 2B-1)
        pair_labels = pair_labels[mask].view(2*B, -1)  # (2B, 2B-1)
        # logits for positive/negative pairs
        logits_pos = sim_matrix[pair_labels == 1].view(2*B, -1)  # (2B, 1)
        logits_neg = sim_matrix[pair_labels == 0].view(2*B, -1)  # (2B, 2(B-1))
        logits = torch.cat([logits_pos, logits_neg], dim=1)  # (2B, 2B-1)
        # apply temperature
        logits /= self.tau

        # (2B-1)-way softmax classification labels
        targets = torch.zeros(2*B, dtype=torch.long, device=features.device)

        return logits, targets


class MoCoModel(nn.Module):
    """Moco model for time series self-supervised learning.

    Referred from https://github.com/facebookresearch/moco.
    """
    def __init__(self, backbone: nn.Module, queue_size=65536, momentum=0.999, tau=0.07, projector=False):
        super(MoCoModel, self).__init__()

        self.queue_size, self.momentum, self.tau = queue_size, momentum, tau
        out_dim = backbone.fc.out_features  # default: 128
        self.backbone = backbone
        self.backbone_k = copy.deepcopy(backbone)

        if projector:  # moco-v2
            mlp_dim = self.backbone[-1].fc.in_features
            self.backbone[-1].fc = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                self.backbone[-1].fc
            )
            self.backbone_k.fc = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                self.backbone_k.fc
            )

        for param, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer('queue', torch.randn(queue_size, out_dim))
        self.queue = F.normalize(self.queue, p=2, dim=1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    def forward(self, x1, x2):
        x_q, x_k = x1, x2

        # compute query features
        q = self.backbone(x_q)
        q = F.normalize(q, p=2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update key encoder
            # shuffle for making use of BatchNorm (only support DDP)
            if is_dist_avail_and_initialized():
                x_k, idx_unshuffle = self._batch_shuffle_ddp(x_k)
            k = self.backbone_k(x_k)
            k = F.normalize(k, p=2, dim=1)
            # undo shuffle (only support DDP)
            if is_dist_avail_and_initialized():
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits: Nx(1+K)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (N, 1); i.e., bmm(q.view(N, 1, C), k.view(N, C, 1))
        l_neg = torch.einsum('nc,kc->nk', [q, self.queue.clone().detach()])  # (N, K); i.e., mm(q.view(N, C), self.queue.view(K, C).T)
        logits = torch.cat([l_pos, l_neg], dim=1)  # (N, 1+K)
        # apply temperature
        logits /= self.tau

        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)  # labels: positive key indicators

        self._dequeue_and_enqueue(k)

        return logits, targets

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + param.data * (1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)  # gather keys before updating queue
        B = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % B == 0, 'Queue size must be divisible by batch size.'  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr+B] = keys
        ptr = (ptr + B) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm. Only support DistributedDataParallel (DDP) model."""
        # gather from all GPUs
        B_local = x.shape[0]
        x = concat_all_gather(x)
        B_global = x.shape[0]
        num_gpus = B_global // B_local

        idx_shuffle = torch.randperm(B_global).cuda()  # random shuffle index
        dist.broadcast(idx_shuffle, src=0)  # broadcast to other GPUs
        idx_unshuffle = torch.argsort(idx_shuffle)  # index for restoring

        # shuffled index for this GPU
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle. Only support DistributedDataParallel (DDP) model."""
        # gather from all GPUs
        B_local = x.shape[0]
        x = concat_all_gather(x)
        B_global = x.shape[0]
        num_gpus = B_global // B_local

        # restore index for this GPUs
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x[idx_this]


class MoCoV3Model(nn.Module):
    """MoCo-v3 model for time series self-supervised learning.

    Referred from https://github.com/facebookresearch/moco-v3.
    """
    def __init__(self, backbone: nn.Module, tau=1.):
        super(MoCoV3Model, self).__init__()

        self.tau = tau
        self.backbone = backbone
        self.momentum_backbone = copy.deepcopy(backbone)

        prev_dim = self.backbone[-1].fc.in_features
        out_dim = self.backbone[-1].fc.out_features  # default: 256
        mlp_dim = out_dim * 16  # default: 4096
        del self.backbone[-1].fc, self.momentum_backbone.fc  # remove the original fc layer
        # projectors
        self.backbone[-1].fc = self._build_mlp(3, prev_dim, mlp_dim, out_dim)
        self.momentum_backbone.fc = self._build_mlp(3, prev_dim, mlp_dim, out_dim)
        # predictor
        self.predictor = self._build_mlp(2, out_dim, mlp_dim, out_dim)  # alternatively, last_bn=False for predictor

        for param, param_m in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def forward(self, x1, x2, m: float):
        # compute predicted features for two views
        q1 = self.predictor(self.backbone(x1))
        q2 = self.predictor(self.backbone(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_backbone(x1)
            k2 = self.momentum_backbone(x2)

        logits1, labels1 = self._compute_logits(q1, k2)
        logits2, labels2 = self._compute_logits(q2, k1)

        return logits1, labels1, logits2, labels2

    @torch.no_grad()
    def _update_momentum_encoder(self, m: float):
        """Momentum update of the momentum encoder."""
        for param, param_m in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            param_m.data = param_m.data * m + param.data * (1 - m)

    def _compute_logits(self, q, k):
        """Compute logits and labels for given query and key."""
        # normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)

        # compute logits and apply temperature
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.tau
        B = logits.shape[0]  # batch size per GPU
        labels = torch.arange(B, dtype=torch.long, device=q.device) + B * dist.get_rank()

        return logits, labels

    def _build_mlp(self, num_layers, in_dim, mlp_dim, out_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = in_dim if l == 0 else mlp_dim
            dim2 = out_dim if l == num_layers-1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers-1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


class BYOLModel(nn.Module):
    """BYOL model for time series self-supervised learning.

    Referred from https://github.com/deepmind/deepmind-research/tree/master/byol.
    """
    def __init__(self, backbone: nn.Module):
        super(BYOLModel, self).__init__()

        self.backbone = backbone
        prev_dim = self.backbone[-1].fc.in_features
        out_dim = self.backbone[-1].fc.out_features  # default: 256
        mlp_dim = out_dim * 16  # default: 4096

        # projector
        """self.backbone[-1].fc = nn.Sequential(
            nn.Linear(prev_dim, mlp_dim, bias=True),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),  # first layer
            nn.Linear(mlp_dim, mlp_dim, bias=True),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),  # second layer
            nn.Linear(mlp_dim, out_dim)  # output layer
        )"""
        self.backbone[-1].fc = nn.Sequential(
            nn.Linear(prev_dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(mlp_dim, out_dim),  # output layer
        )
        self.momentum_backbone = copy.deepcopy(backbone)

        """self.predictor = nn.Sequential(
            nn.Linear(out_dim, mlp_dim, bias=True),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),  # first layer
            nn.Linear(mlp_dim, out_dim)  # output layer
        )"""
        # predictor: same architecture as the projector
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(mlp_dim, out_dim),  # output layer
        )

        for param, param_m in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def forward(self, x1, x2, m: float):
        # compute predicted features for two views
        p1 = self.predictor(self.backbone(x1))
        p2 = self.predictor(self.backbone(x2))
        p1, p2 = F.normalize(p1, dim=1), F.normalize(p2, dim=1)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            z1 = self.momentum_backbone(x1)
            z2 = self.momentum_backbone(x2)
            z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)

        return p1, p2, z1.detach(), z2.detach()

    @torch.no_grad()
    def _update_momentum_encoder(self, m: float):
        """Momentum update of the momentum encoder."""
        for param, param_m in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            param_m.data = param_m.data * m + param.data * (1 - m)


class SimSiamModel(nn.Module):
    """SimSiam model for time series self-supervised learning.

    Referred from https://github.com/facebookresearch/simsiam.
    """
    def __init__(self, backbone: nn.Module):
        super(SimSiamModel, self).__init__()

        self.backbone = backbone
        prev_dim = self.backbone[-1].fc.in_features
        out_dim = self.backbone[-1].fc.out_features  # default: 2048
        mlp_dim = out_dim // 4  # default: 512; bottleneck structure for the predictor

        # 3-layer projection head
        self.backbone[-1].fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.backbone[-1].fc,
            nn.BatchNorm1d(out_dim, affine=False)  # output layer
        )
        self.backbone[-1].fc[6].bias.requires_grad = False  # not use bias as it's followed by BN

        # 2-layer prediction head
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(mlp_dim, out_dim)  # output layer
        )

    def forward(self, x1, x2):
        # compute features for one view
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)

        # predict the other view
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()
