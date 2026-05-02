#!/usr/bin/env python
from __future__ import print_function
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import (confusion_matrix, classification_report,
                              balanced_accuracy_score, roc_auc_score)
import csv
import numpy as np
import glob
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
# from cosine_lr_schedueler import CosineLRScheduler
# import resource
import copy
from torch import linalg as LA


"https://github.com/ajbrock/BigGAN-PyTorch/blob/master/utils.py"

class DictAction(argparse.Action):
    """
    argparse action untuk menerima argumen dict dari command line.
    Format: --arg key1=val1 key2=val2
    """
    @staticmethod
    def _parse_value(v):
        if v.lower() == 'true':
            return True
        if v.lower() == 'false':
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    def __call__(self, parser, namespace, values, option_string=None):
        result = getattr(namespace, self.dest, None) or {}
        if isinstance(values, list):
            for item in values:
                if '=' in item:
                    k, v = item.split('=', 1)
                    result[k.strip()] = self._parse_value(v.strip())
                else:
                    raise argparse.ArgumentTypeError(
                        "Format harus key=value, dapat: '{}'".format(item))
        setattr(namespace, self.dest, result)


def ema_update(source, target, decay=0.99, start_itr=20, itr=None):
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.
    if itr and itr<start_itr:
        decay = 0.0
    with torch.no_grad():
        for key, value in source.state_dict().items():
            target.state_dict()[key].copy_(target.state_dict()[key] * decay + value * (1 - decay))


def build_loss(loss_name: str, loss_args: dict) -> nn.Module:
    args = dict(loss_args) if loss_args else {}
    if 'weight' in args:
        w = args['weight']
        if w is not None and not isinstance(w, torch.Tensor):
            args['weight'] = torch.tensor(w, dtype=torch.float32)
        elif w is None:
            del args['weight']
    LossClass = getattr(nn, loss_name, None)
    if LossClass is None:
        raise ValueError("Loss '{}' tidak ditemukan di torch.nn".format(loss_name))
    return LossClass(**args)


def move_loss_to_device(loss: nn.Module, device) -> nn.Module:
    return loss.to(device)


try:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
except ImportError:
    pass  # Windows tidak punya resource module

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    # training speed is too slow if set to True
    torch.backends.cudnn.deterministic = True

    # on cuda 11 cudnn8, the default algorithm is very slow
    # unlike on cuda 10, the default works well
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_mmd_loss(z, z_prior, y, num_cls):
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y == i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean = LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean, z_mean[y_valid]


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='GRA Transformer')

    parser.add_argument('--work-dir', default='./work_dir/temp',
                        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--config', default='./config/nturgbd-cross-view/test_bone.yaml',
                        help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=True, help='if ture, the classification score will be stored')

    # gra
    parser.add_argument(
        '--joint-label',
        type=list,
        default=[],
        help='tells which group each joint belongs to')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=10,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--ema',
        action="store_true",
        default=False,
        help='ema weight for eval')

    parser.add_argument('--lambda_1', type=float, default=1e-4)
    parser.add_argument('--lambda_2', type=float, default=1e-1)

    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=4,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=[0,1,2,3,4,5,6,7],
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='nesterov momentum')
    parser.add_argument(
        '--alpha', type=bool, default=False, help='coefficient for statistical topology encoding')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument(
        '--use_weighted_sampler',
        type=str2bool,
        default=False,
        help='gunakan WeightedRandomSampler untuk menangani class imbalance')

    # loss
    parser.add_argument('--loss', default='CrossEntropyLoss')
    parser.add_argument('--loss-args', action=DictAction, default=dict())

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.best_f1 = 0.0
        self.best_f1_epoch = 0

        self.best_acc_ema = 0
        self.best_acc_epoch_ema = 0

        self.use_cuda = torch.cuda.is_available()
        self.model = self.model.cuda(self.output_device)
        self.loss = move_loss_to_device(self.loss, self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
                self.loss = move_loss_to_device(self.loss, self.output_device)
        if self.arg.ema:
            Model = import_class(self.arg.model)
            self.model_ema = Model(**self.arg.model_args).cuda(self.output_device)
            ema_update(self.model, self.model_ema, itr=0)


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            train_dataset = Feeder(**self.arg.train_feeder_args)
            use_weighted  = getattr(self.arg, 'use_weighted_sampler', False)
            sampler       = (train_dataset.get_weighted_sampler()
                             if use_weighted else None)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
                pin_memory=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        print(self.model)
        self.loss = build_loss(
            getattr(self.arg, 'loss', 'CrossEntropyLoss'),
            getattr(self.arg, 'loss_args', {})
        )

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':

            # my_list = ['fc.weight', 'fc.bias', 'fc_mu.weight', 'fc_mu.bias','fc_logvar.weight', 'fc_logvar.bias', 'decoder.weight', 'decoder.bias']
            # params = list(filter(lambda kv: kv[0] in my_list, self.model.named_parameters()))
            # base_params = list(filter(lambda kv: kv[0] not in my_list, self.model.named_parameters()))
            #
            # params = [i[1] for i in params]
            # base_params = [i[1] for i in base_params]

            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=self.arg.momentum,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'NAdam':
            self.optimizer = optim.NAdam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer in ('SGD', 'Adam', 'NAdam', 'AdamW'):
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        if self.arg.ema:
            self.model_ema.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        use_amp = self.use_cuda
        amp_device = 'cuda' if self.use_cuda else 'cpu'
        scaler = torch.amp.GradScaler(amp_device, init_scale=256, enabled=use_amp)

        num_class = (self.model.module.num_class
                     if hasattr(self.model, "module") else self.model.num_class)

        for batch_idx, (data, label) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().to(self.output_device)
                label = label.long().to(self.output_device)
            timer['dataloader'] += self.split_time()

            with torch.amp.autocast(amp_device, enabled=use_amp):
                output = self.model(data)
                loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20.0)
            scaler.step(self.optimizer)
            scaler.update()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(
                np.mean(loss_value), np.mean(acc_value) * 100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        if self.arg.ema:
            self.model_ema.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        best_f1_this_eval = 0.0
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []

            loss_value_ema = []
            score_frag_ema = []
            label_list_ema = []
            pred_list_ema = []
            step = 0
            num_class = (self.model.module.num_class
                         if hasattr(self.model, "module") else self.model.num_class)
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label) in enumerate(process):
                label_list.append(label)
                if self.arg.ema:
                    label_list_ema.append(label)
                with torch.no_grad():
                    data = data.float().to(self.output_device)
                    label = label.long().to(self.output_device)

                    output = self.model(data)

                    if self.arg.ema:
                        self.model_ema.cuda(self.output_device)
                        output_ema = self.model_ema(data)
                    loss = self.loss(output, label)
                    if self.arg.ema:
                        loss_ema = self.loss(output_ema, label)

                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    if self.arg.ema:
                        score_frag_ema.append(output_ema.data.cpu().numpy())
                        loss_value_ema.append(loss_ema.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())

                    if self.arg.ema:
                        _, predict_label_ema = torch.max(output_ema.data, 1)
                        pred_list_ema.append(predict_label_ema.data.cpu().numpy())

                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            sample_idx = batch_idx * len(true) + i
                            f_w.write(str(sample_idx) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)

            if self.arg.ema:
                score_ema = np.concatenate(score_frag_ema)
                loss_ema = np.mean(loss_value_ema)

            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)

            if self.arg.ema:
                accuracy_ema = self.data_loader[ln].dataset.top_k(score_ema, 1)

            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            if self.arg.ema:
                if accuracy_ema > self.best_acc_ema:
                    self.best_acc_ema = accuracy_ema
                    self.best_acc_epoch_ema = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.ema:
                print('Accuracy_ema: ', accuracy, ' model_ema: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)
                if self.arg.ema:
                    self.val_writer.add_scalar('loss_ema', loss_ema, self.global_step)
                    self.val_writer.add_scalar('acc_ema', accuracy_ema, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            if self.arg.ema:
                score_dict_ema = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score_ema))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            if self.arg.ema:
                self.print_log('\tMean {} loss_ema of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value_ema)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
            if self.arg.ema:
                for k in self.arg.show_topk:
                    self.print_log('\tTop{}_ema: {:.2f}%'.format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score_ema, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            if self.arg.ema:
                if save_score:
                    with open('{}/epoch{}_{}_score_ema.pkl'.format(
                            self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                        pickle.dump(score_dict_ema, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

            # ---- Comprehensive classification metrics ----
            label_names = (['non-fall', 'fall'] if num_class == 2
                           else [str(i) for i in range(num_class)])
            report = classification_report(
                label_list, pred_list, target_names=label_names, zero_division=0)
            self.print_log('\n  Classification Report ({}):\n{}'.format(ln, report))

            bal_acc = balanced_accuracy_score(label_list, pred_list)
            self.print_log('\tBalanced Accuracy: {:.2f}%'.format(bal_acc * 100))

            if num_class == 2 and confusion.shape == (2, 2):
                tn, fp, fn, tp = confusion.ravel()
                prec = tp / max(tp + fp, 1)
                rec  = tp / max(tp + fn, 1)
                spec = tn / max(tn + fp, 1)
                f1   = 2 * prec * rec / max(prec + rec, 1e-9)
                best_f1_this_eval = max(best_f1_this_eval, f1)

                self.print_log('\tSensitivity (Fall Recall)    : {:.2f}%'.format(rec  * 100))
                self.print_log('\tSpecificity (Non-Fall Recall): {:.2f}%'.format(spec * 100))
                self.print_log('\tTP={:4d}  TN={:4d}  FP={:4d}  FN={:4d}'.format(
                    int(tp), int(tn), int(fp), int(fn)))

                auc = float('nan')
                try:
                    score_proba = F.softmax(
                        torch.from_numpy(score.astype(np.float32)), dim=1).numpy()
                    auc = roc_auc_score(label_list, score_proba[:, 1])
                    self.print_log('\tAUC-ROC: {:.4f}'.format(auc))
                except Exception as e:
                    self.print_log('\tAUC-ROC: N/A ({})'.format(e))

                if self.arg.phase == 'train':
                    self.val_writer.add_scalar('precision',    prec,    self.global_step)
                    self.val_writer.add_scalar('recall',       rec,     self.global_step)
                    self.val_writer.add_scalar('f1',           f1,      self.global_step)
                    self.val_writer.add_scalar('balanced_acc', bal_acc, self.global_step)
                    self.val_writer.add_scalar('auc_roc',      auc,     self.global_step)

                with open('{}/epoch{}_{}_binary_metrics.csv'.format(
                        self.arg.work_dir, epoch + 1, ln), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['metric', 'value'])
                    writer.writerow(['accuracy',          '{:.4f}'.format(accuracy)])
                    writer.writerow(['balanced_accuracy', '{:.4f}'.format(bal_acc)])
                    writer.writerow(['precision',         '{:.4f}'.format(prec)])
                    writer.writerow(['sensitivity',       '{:.4f}'.format(rec)])
                    writer.writerow(['specificity',       '{:.4f}'.format(spec)])
                    writer.writerow(['f1',                '{:.4f}'.format(f1)])
                    writer.writerow(['auc_roc',           '{:.4f}'.format(auc)])
                    writer.writerow(['tp', int(tp)])
                    writer.writerow(['tn', int(tn)])
                    writer.writerow(['fp', int(fp)])
                    writer.writerow(['fn', int(fn)])
            # ---- End comprehensive metrics ----

            if self.arg.ema:
                # acc for each class:
                label_list_ema = np.concatenate(label_list_ema)
                pred_list_ema = np.concatenate(pred_list_ema)
                confusion_ema = confusion_matrix(label_list_ema, pred_list_ema)
                list_diag_ema = np.diag(confusion_ema)
                list_raw_sum_ema = np.sum(confusion_ema, axis=1)
                each_acc_ema = list_diag_ema / list_raw_sum_ema
                with open('{}/epoch{}_{}_each_class_acc_ema.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(each_acc_ema)
                    writer.writerows(confusion_ema)

        if wrong_file is not None and f_w is not None:
            f_w.close()
        if result_file is not None and f_r is not None:
            f_r.close()
        return best_f1_this_eval

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                if self.arg.ema:
                    ema_update(self.model, self.model_ema, itr=epoch)
                epoch_f1 = self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
                if epoch_f1 > self.best_f1:
                    self.best_f1       = epoch_f1
                    self.best_f1_epoch = epoch + 1

                # Selalu simpan kalau ini best accuracy baru
                if self.best_acc_epoch == (epoch + 1) and not save_model:
                    state_dict = self.model.state_dict()
                    weights = OrderedDict([
                        [k.split('module.')[-1], v.cpu()]
                        for k, v in state_dict.items()
                    ])
                    torch.save(weights,
                               self.arg.model_saved_name + '-' + str(epoch + 1) +
                               '-' + str(int(self.global_step)) + '.pt')

            # test the best model
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            pattern = os.path.join(self.arg.work_dir, 'runs-' + str(self.best_acc_epoch) + '*')
            best_files = glob.glob(pattern)
            if best_files:
                weights_path = best_files[0]
                map_loc = ('cuda:{}'.format(self.output_device)
                           if self.use_cuda else 'cpu')
                weights = torch.load(weights_path, map_location=map_loc,
                                     weights_only=False)
                if type(self.arg.device) is list and len(self.arg.device) > 1:
                    weights = OrderedDict([
                        ['module.' + k, v.to(self.output_device)]
                        for k, v in weights.items()
                    ])
                self.model.load_state_dict(weights)

                wf = weights_path.replace('.pt', '_wrong.txt')
                rf = weights_path.replace('.pt', '_right.txt')
                self.print_log('\n' + '='*60)
                self.print_log('Final evaluation on best model (epoch {})'.format(self.best_acc_epoch))
                self.print_log('='*60)
                self.arg.print_log = False
                self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
                self.arg.print_log = True

            self.print_log('\n' + '='*60)
            self.print_log('Training complete. Summary:')
            self.print_log('  Best accuracy     : {:.2f}%'.format(self.best_acc * 100))
            self.print_log('  Best acc epoch    : {}'.format(self.best_acc_epoch))
            self.print_log('  Best F1           : {:.4f}'.format(self.best_f1))
            self.print_log('  Best F1 epoch     : {}'.format(self.best_f1_epoch))
            self.print_log('  Model dir         : {}'.format(self.arg.work_dir))
            self.print_log('  Trainable params  : {:,}'.format(num_params))
            self.print_log('  Weight decay      : {}'.format(self.arg.weight_decay))
            self.print_log('  Base LR           : {}'.format(self.arg.base_lr))
            self.print_log('  Batch size        : {}'.format(self.arg.batch_size))
            self.print_log('  Seed              : {}'.format(self.arg.seed))

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        valid_keys = set(vars(p).keys())
        normalized_arg = {}
        for k, v in default_arg.items():
            k_norm = k.replace('-', '_')
            if k_norm not in valid_keys:
                print('WRONG ARG (diabaikan): {}'.format(k))
                continue
            normalized_arg[k_norm] = v
        parser.set_defaults(**normalized_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()