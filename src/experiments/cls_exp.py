from .base_exp import *
from src.models.build import get_model
from src.models.encoders.build import get_encoder


class ClassificationExp(BaseExp):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.model = self.build_model()
        self.get_optimizer()
        self.get_scheduler()
        save_model_architecture(config, self.model)

    def build_model(self):
        # get encoder and backbone
        encoder = get_encoder(self.config, self.dataloader_dict['train'].dataset)
        model = get_model(self.config)
        model = nn.Sequential(encoder, model)
        # load pretrained model and freeze backbone
        if self.config.pretrained_model:
            model = self.load_pretrained_model(model)
        if self.config.freeze_backbone:
            model = self.freeze_backbone(model)
        # model parallel
        if self.config.gpu is None:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda(self.config.gpu)

        return model

    def load_pretrained_model(self, model):
        self.logger.info(f'Loading the {self.config.checkpoint_mode} checkpoint of pretrained model from {self.config.load_checkpoint_path}...')
        checkpoint = torch.load(self.config.load_checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        self.logger.info('Renaming parameters and removing parameters of fc layer...')
        for k in list(state_dict.keys()):
            if k.startswith('backbone.') and not k.startswith('backbone.fc.'):
                state_dict[k[len('backbone.'):]] = state_dict[k]  # remove prefix 'backbone.'
            if k.startswith('backbone.') or k.startswith('fc.'):
                del state_dict[k]  # remove old-named and fc parameters

        model_without_ddp = model.module if hasattr(model, 'module') else model
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        # print(msg)

        # interpolate learnable position encodings for transformer
        # if 'wint' in self.config.model:
        #     num_pos = self.config.model_series_size // self.config.window_size + 1
        #     if num_pos != model_without_ddp.pos_embed.shape[1]:  # interpolation is neccessary
        #         self.logger.info(f'Interpolating the position embedding from {model_without_ddp.pos_embed.shape[1]} to {num_pos}...')
        #         window_pe = F.interpolate(model_without_ddp.pos_embed.transpose(1, 2)[:, :, 1:], size=num_windows, mode='linear', align_corners=False).transpose(1, 2)
        #         new_pe = torch.cat([model_without_ddp.pos_embed[:, :1, :], window_pe], dim=1)
        #         model_without_ddp.pos_embed.data = new_pe  # type of model.pos_embed is torch.nn.Parameter

        return model_without_ddp

    def freeze_backbone(self, model):
        # freeze parameters of backbone (i.e., except the last fc layer)
        self.logger.info('Freezing the parameters of backbone...')
        for name, param in model.named_parameters():
            if name not in ['1.fc.weight', '1.fc.bias']:
                param.requires_grad = False
            elif 'weight' in name:
                param.data.normal_(mean=0., std=0.01)
            elif 'bias' in name:
                param.data.zero_()
        update_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(update_parameters) == 2, f'Only the 2 parameters in fc layer are supposed to update, but get {len(update_parameters)} parameters!'

        return model

    @torch.no_grad()
    def evaluate(self, type):
        iter_time, data_time = AverageMeter(), AverageMeter()
        Losses, Acc1s, Acc5s = AverageMeter(), AverageMeter(), AverageMeter()
        # switch to evaluation mode
        self.model.eval()
        # prediction results
        logits_tensor = torch.tensor([], dtype=torch.float)
        preds_tensor = torch.tensor([], dtype=torch.long)
        trues_tensor = torch.tensor([], dtype=torch.long)
        if torch.cuda.is_available():
            logits_tensor = logits_tensor.cuda(self.config.gpu)
            preds_tensor = preds_tensor.cuda(self.config.gpu)
            trues_tensor = trues_tensor.cuda(self.config.gpu)

        # iterate on evaluation dataset
        torch.cuda.synchronize()
        start_time = time.time()
        for (samples, targets) in self.dataloader_dict[type]:
            B = samples.shape[0]  # batch size
            if torch.cuda.is_available():
                samples = samples.cuda(self.config.gpu)
                targets = targets.cuda(self.config.gpu)

            logits = self.model(samples)
            loss = self.loss(logits, targets)
            Losses.update(loss.item(), B)
            # measure accuracy
            acc1, acc5 = accuracy(logits, targets, topk=(1, min(5, self.config.num_classes//2)))
            Acc1s.update(acc1, B)
            Acc5s.update(acc5, B)

            # accuracy per class
            _, preds = torch.max(logits, dim=1)
            logits_tensor = torch.cat([logits_tensor, logits.float()], dim=0)
            preds_tensor = torch.cat([preds_tensor, preds.long()], dim=0)
            trues_tensor = torch.cat([trues_tensor, targets.long()], dim=0)

            # measure elapsed time
            torch.cuda.synchronize()
            iter_time.update(time.time() - start_time)
            start_time = time.time()

        # logits_tensor = concat_all_gather(logits_tensor)
        preds_tensor = concat_all_gather(preds_tensor)
        trues_tensor = concat_all_gather(trues_tensor)

        # save result
        val_acc = Acc1s.avg
        _, val_mf1 = save_result(self.config, type, logits_tensor, preds_tensor, trues_tensor, verbose=False)

        return val_acc, val_mf1

    def train_batch(self, epoch, idx, batch_data):
        samples, targets = batch_data
        B = samples.shape[0]
        if torch.cuda.is_available():
            samples = samples.cuda(self.config.gpu)
            targets = targets.cuda(self.config.gpu)
        # compute output and loss
        logits = self.model(samples)
        loss = self.loss(logits, targets)
        acc1, acc5 = accuracy(logits, targets, topk=(1, min(5, self.config.num_classes//2)))

        return B, loss, acc1, acc5
