import time
import torch

from src.data.dataloader import get_dataloader
from src.utils.util import *


class BaseExp:
    """Define the basic APIs"""

    def __init__(self, config, logger, device='cuda'):
        super().__init__()
        self.config = config
        self.logger = logger
        self.device = device

        self.prepare_data()
        self.prepare_loss()

    def prepare_loss(self):
        self.loss = get_loss(self.config)

    def prepare_data(self):
        self.logger.info(f'Preparing data')
        self.dataloader_dict = get_dataloader(self.config)

    def get_optimizer(self):
        self.optimizer = get_optimizer(self.config, self.model)

    def get_scheduler(self):
        self.scheduler = get_scheduler(self.config, self.optimizer)

    # optionally resume from a checkpoint
    def load_checkpoint(self):
        start_epoch, best_val_acc = 0, 0.0
        if self.config.resume:
            if self.config.gpu is None:
                checkpoint = torch.load(self.config.load_checkpoint_path)
            else:
                checkpoint = torch.load(self.config.load_checkpoint_path, map_location=f'cuda:{self.config.gpu}')
            start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f'Loading checkpoint at epoch {start_epoch-1} from {self.config.load_checkpoint_path}...')
            model_without_ddp = self.model.module if hasattr(self.model, 'module') else self.model
            model_without_ddp.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            best_val_acc = checkpoint['best_val_acc']
            best_val_mf1 = checkpoint.get('best_val_mf1', 0.0)

        return start_epoch, best_val_acc

    def save_checkpoint(self, epoch, val_acc, val_mf1=0.0, test_acc=0.0, test_mf1=0.0):
        if self.config.use_eval:
            self.logger.info(
                f'Evaluation: '
                f'Val: {val_acc:.3f}% | {val_mf1:.3f}%, '
                f'Test: {test_acc:.3f}% | {test_mf1:.3f}% '
            )
        if self.config.save_checkpoint:
            model_without_ddp = self.model.module if hasattr(self.model, 'module') else self.model
            save_dict = {
                'epoch': epoch,
                'state_dict': model_without_ddp.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_val_acc': val_acc,
                'best_val_mf1': val_mf1,
                'best_test_acc': test_acc,
                'best_test_mf1': test_mf1
            }
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)
            # For cls task, only save best checkpoint and for ssl save with freq
            if val_acc > 0:
                torch.save(save_dict, os.path.join(self.config.output_dir, 'checkpoint_best.pth'))
            elif epoch % self.config.ssl_save_checkpoint_freq == 0:
                torch.save(save_dict, os.path.join(self.config.output_dir, f'checkpoint_{epoch}.pth'))
        # self.logger.info(f'Saving checkpoint to {self.config.checkpoint_dir}/checkpoint_best.pth...')

    def train_epoch(self, epoch):
        # statics
        iter_time, data_time = AverageMeter(), AverageMeter()
        Losses, Acc1s, Acc5s = AverageMeter(), AverageMeter(), AverageMeter()
        # model status
        self.model.train()

        start_time = time.time()
        for idx, batch_data in enumerate(self.dataloader_dict["train"]):
            torch.cuda.synchronize()
            data_time.update(time.time() - start_time)
            # train
            B, loss, acc1, acc5 = self.train_batch(epoch, idx, batch_data)
            Acc1s.update(acc1, B)
            Acc5s.update(acc5, B)
            Losses.update(loss.item(), B)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if not self.config.t_in_epochs:
                self.scheduler.step_update(epoch * self.config.iters_per_epoch + idx)
            # measure time
            torch.cuda.synchronize()
            iter_time.update(time.time() - start_time)
            start_time = time.time()

        # print epoch info
        self.logger.info(
            f'Training Epoch [{epoch:3d}] '
            f'LR: {self.optimizer.param_groups[-1]["lr"]:.6f} '
            f'Batch Time {iter_time.sum:.3f}s '
            f'Data Time {data_time.sum:.3f}s '
            f'Loss {Losses.avg:.6f} '
            f'Acc@1 {Acc1s.avg:.3f}%' 
        )

    def run(self):
        # load checkpoint
        start_epoch, best_val_acc = self.load_checkpoint()
        self.logger.info(
            f'Start training from epoch {start_epoch}...'
            f'[{self.config.dataset}] Training data size: {len(self.dataloader_dict["train"].dataset)}; '
            f'Training data iterations per epoch: {self.config.iters_per_epoch}; '
            f'Validation data size: {len(self.dataloader_dict["val"].dataset)}; '
            f'Test data size: {len(self.dataloader_dict["test"].dataset)}; '
            f'Original Series length: {self.config.ori_series_size}; '
            f'Model Series length: {self.config.model_series_size}; '
            f'Number of classes: {self.config.num_classes}; '
            f'Number of channels: {self.config.num_channels}'
        )

        early_stopping = EarlyStopping(patience=self.config.patience, best_score=best_val_acc)
        start_epoch, best_epoch = 0, 0
        best_val_acc, best_val_mf1 = 0., 0.
        best_test_acc, best_test_mf1  = 0., 0.
        for epoch in range(start_epoch, self.config.num_epochs):
            self.train_epoch(epoch)
            if self.config.task == 'ssl' and not self.config.use_eval:
                self.save_checkpoint(epoch, best_val_acc, best_val_mf1, best_test_acc, best_test_mf1)
            # without earlystop
            # val_acc, val_mf1 = self.evaluate('val')
            # best_val_acc, best_val_mf1 = val_acc, val_mf1
            # best_test_acc, best_test_mf1 = val_acc, val_mf1
            # self.save_checkpoint(epoch, best_val_acc, best_val_mf1, best_test_acc, best_test_mf1)
            elif epoch % self.config.val_freq == 0:
                val_acc, val_mf1 = self.evaluate('val')
                early_stopping(val_acc)
                if early_stopping.counter == 0:
                    best_epoch = epoch
                    best_val_acc, best_val_mf1 = val_acc, val_mf1
                    # val is test
                    # if self.config.no_validation_set:
                    #     best_test_acc, best_test_mf1 = val_acc, val_mf1
                    # else:
                    best_test_acc, best_test_mf1 = self.evaluate('test')
                    self.save_checkpoint(epoch, best_val_acc, best_val_mf1, best_test_acc, best_test_mf1)
                if early_stopping.early_stop:
                    self.logger.info(f'Early stopping')
                    break

            # step learning rate for next epoch
            if self.config.t_in_epochs:
                self.scheduler.step(epoch=epoch+1, metric=val_acc)

        self.logger.info(
            'Training finished! '
            f'Best Acc | Macro-F1 on val: {best_val_acc:.3f}% | {best_val_mf1:.3f}%, '
            f'test: {best_test_acc:.3f}% | {best_test_mf1:.3f}% '
            f'(on epoch {best_epoch:3d}/{self.config.num_epochs:3d})')

        return best_test_acc, best_test_mf1
