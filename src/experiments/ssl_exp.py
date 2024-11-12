import math
import torch.nn.functional as F

from .base_exp import *
from src.models.build import get_model, get_ssl_model
from src.models.encoders.build import get_encoder


class SelfSupervisedLearningExp(BaseExp):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.model = self.build_model()
        self.get_optimizer()
        self.get_scheduler()
        save_model_architecture(config, self.model)

    def build_model(self):
        # get encoder
        encoder = get_encoder(self.config, self.dataloader_dict['train'].dataset)
        model = get_model(self.config)
        # encoder + model as backbone
        model = nn.Sequential(encoder, model)
        model = get_ssl_model(self.config, backbone=model)
        if self.config.gpu is None:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda(self.config.gpu)

        return model

    def kNN(self, type):
        """kNN evaluation process."""
        # load all training data with transformations same as testing data
        pre_time, net_time, cls_time = AverageMeter(), AverageMeter(), AverageMeter()
        Acc1s, Acc5s = AverageMeter(), AverageMeter()
        # switch to evaluation mode
        self.model.eval()
        model_without_ddp = self.model.module if hasattr(self.model, 'module') else self.model
        dim_features = model_without_ddp.backbone[-1].fc[0].in_features
        # compute features for training data
        train_features = torch.randn(dim_features, len(self.dataloader_dict['train_knn'].dataset))
        torch.cuda.synchronize()  # waits for all kernels in all streams on a CUDA device to complete
        start_time = time.time()
        for idx, (samples, targets) in enumerate(self.dataloader_dict['train_knn']):
            # compute features
            B = samples.shape[0]
            if torch.cuda.is_available():
                samples = samples.cuda(self.config.gpu)
            _ = model_without_ddp.backbone(samples)
            features = model_without_ddp.backbone[-1].features.detach()
            train_features[:, idx*B:(idx+1)*B] = features.T.cpu()
            # measure elapsed time
            torch.cuda.synchronize()
            pre_time.update(time.time() - start_time)
            start_time = time.time()
        train_features = F.normalize(train_features, p=2, dim=0)  # normalize features
        train_targets = torch.LongTensor(self.dataloader_dict['train_knn'].dataset.targets)
        # self.logger.info(f'Memory bank in shape {list(train_features.shape)} has been ready')

        # prediction results
        preds_tensor = torch.tensor([], dtype=torch.long)
        trues_tensor = torch.tensor([], dtype=torch.long)
        retrieval_one_hot = torch.zeros(self.config.k, self.config.num_classes)
        test_features = torch.randn(len(self.dataloader_dict[type].dataset), dim_features)
        # iterate on evaluation dataset
        torch.cuda.synchronize()
        start_time = time.time()
        for idx, (samples, targets) in enumerate(self.dataloader_dict[type]):
            # compute features
            B = samples.shape[0]  # batch size
            C = self.config.num_classes  # number of classes

            if torch.cuda.is_available():
                samples = samples.cuda(self.config.gpu)
            _ = model_without_ddp.backbone(samples)
            features = model_without_ddp.backbone[-1].features.detach().cpu()
            features = F.normalize(features, p=2, dim=1)  # normalize features
            test_features[idx*B:(idx+1)*B] = features

            # measure forward time
            torch.cuda.synchronize()
            net_time.update(time.time() - start_time)

            # kNN
            dist = torch.mm(features, train_features)

            yd, yi = dist.topk(self.config.k, dim=1, largest=True, sorted=True) # (B, k) of distance, (B, k) of index
            candidates = train_targets.view(1, -1).expand(B, -1)
            retrieval = torch.gather(input=candidates, dim=1, index=yi) # (B, k) of targets

            retrieval_one_hot.resize_(B*self.config.k, C).zero_()
            retrieval_one_hot.scatter_(dim=1, index=retrieval.view(-1, 1), value=1)
            yd_transform = yd.clone().div_(self.config.tau).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(B, -1, C), yd_transform.view(B, -1, 1)), dim=1)  # (B, k, C) of dist_sum
            _, predictions = probs.sort(dim=1, descending=True) # (B, C), get class prob by sorted dist

            correct = predictions.eq(targets.view(-1, 1))
            acc1 = correct.narrow(dim=1, start=0, length=1).sum().item() / B * 100.
            acc5 = correct.narrow(dim=1, start=0, length=min(min(self.config.num_classes//2, 5), self.config.k)).sum().item() / B * 100.
            Acc1s.update(acc1, B)
            Acc5s.update(acc5, B)

            preds_tensor = torch.cat([preds_tensor, predictions[:, 0].long()], dim=0)
            trues_tensor = torch.cat([trues_tensor, targets.long()], dim=0)

            # measure elapsed time
            torch.cuda.synchronize()
            cls_time.update(time.time() - start_time)
            start_time = time.time()

        # save result
        val_acc = Acc1s.avg
        val_mf1 = 0.0
        # _, val_mf1 = save_result(self.config, preds_tensor, trues_tensor, verbose=False)

        # logging
        self.logger.info(
            f'{self.config.k}-NN Evaluation '
            f'Preparation Time: {pre_time.sum:.3f}s '
            f'Forward Time: {net_time.sum:.3f}s '
            f'kNN Time: {cls_time.sum:.3f}s '
            f'Acc@1 {val_acc:.3f}% '
            f'Acc@{min(self.config.num_classes//2, 5)} {Acc5s.avg:.3f}% '
            f'Marco F1 {val_mf1:.3f}%'
        )

        return val_acc, val_mf1

    @torch.no_grad()
    def evaluate(self, type):
        acc, mf1 = self.kNN(type=type)

        return acc, mf1

    def train_batch(self, epoch, idx, batch_data):
        samples, targets = batch_data
        sample1, sample2 = samples
        B = sample1.shape[0]  # batch size
        if torch.cuda.is_available():
            sample1 = sample1.cuda(self.config.gpu)
            sample2 = sample2.cuda(self.config.gpu)
        # ssl_method
        acc1, acc5 = 0.0, 0.0
        if self.config.ssl_method in ['simclr', 'moco-v1', 'moco-v2']:  # instance discrimination
            logits, targets = self.model(sample1, sample2)
            loss = self.loss(logits, targets)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        elif self.config.ssl_method == 'moco-v3':  # instance discrimination & symmetric loss
            logits1, targets1, logits2, targets2 = self.model(sample1, sample2, self.config.moco_momentum)
            loss = (self.loss(logits1, targets1) + self.loss(logits2, targets2)) * 2 * self.config.tau
            acc1, acc5 = (accuracy(logits1, targets1, topk=(1, 5)) + accuracy(logits2, targets2, topk=(1, 5))) / 2
        elif self.config.ssl_method in ['byol', 'simsiam']:  # prediction & symmetric loss
            if self.config.ssl_method == 'byol':  # moving average encoder
                if self.config.cos_moco_m:  # gradually increase the moco momentum to 1 with a half-cycle cosine schedule
                    ep = epoch + idx / self.config.iters_per_epoch
                    m = 1. - 0.5 * (1. + math.cos(math.pi * ep / self.config.num_epochs)) * (1. - self.config.moco_momentum)
                else:  # use the same momentum for all epochs
                    m = self.config.moco_momentum
                p1, p2, z1, z2 = self.model(sample1, sample2, m)
            else:
                p1, p2, z1, z2 = self.model(sample1, sample2)
            loss = (self.loss(p1, z2) + self.loss(p2, z1)).mean()
            if self.config.ssl_method == 'simsiam': loss *= -1  # simsiam: maximize cosine similarity

        # print epoch info
        if idx % self.config.ssl_batch_log_freq == 0:
            self.logger.info(
                f'Training Epoch [{epoch:3d}] '
                f'Itr: {idx}/{self.config.iters_per_epoch} '
                f'Loss: {loss:.6f} '
            )

        return B, loss, acc1, acc5
