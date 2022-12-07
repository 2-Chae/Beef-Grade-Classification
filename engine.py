# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torch.nn as nn
import utils
import sklearn
from sklearn.metrics import confusion_matrix, cohen_kappa_score

import ttach as tta
import pickle


def prediction2label(pred, apply_sigmoid=False):
    """Convert ordinal predictions to class labels, e.g.
    
    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    if apply_sigmoid:
        pred = pred.sigmoid().cpu().numpy()
    else:
        pred = pred.cpu().numpy()
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1

class OrdinalRegressionLoss(nn.Module):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""
    def __init__(self):
        super(OrdinalRegressionLoss, self).__init__()
    
    def forward(self, pred, targets):
        
        pred = pred.sigmoid()

        modified_target = torch.zeros_like(pred)

        for i, target in enumerate(targets):
            modified_target[i, 0:target+1] = 1
    
        return nn.MSELoss(reduction='none')(pred, modified_target).sum(axis=1).mean()

class GaussianLabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(GaussianLabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        # print(pred)

        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            prev_idx = target.data.unsqueeze(1)-1
            next_idx = target.data.unsqueeze(1)+1

            prev_idx[prev_idx < 0] = 0
            next_idx[next_idx > 4] = 4

            true_dist.scatter_(1, prev_idx, self.smoothing / 2)
            true_dist.scatter_(1, next_idx, self.smoothing / 2)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    gt = []
    pred = []

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        gt.extend(target.tolist())
        _, predicted = output.max(1)
        pred.extend(predicted.tolist())
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=None)
    print("cohen kappa score: %f" % score)

    sample_weight = sklearn.utils.class_weight.compute_sample_weight(class_weight="balanced", y=gt, indices=None)
    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=sample_weight)
    print("cohen kappa score with sample weight: %f" % score)

    metric_logger.meters['ck'].update(score, n=len(data_loader))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} CK_score {ck.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, ck=metric_logger.ck, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, pred

@torch.no_grad()
def evaluate_reversed(data_loader, model, device, use_amp=False):
    criterion = OrdinalRegressionLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    gt = []
    pred = []

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)


        #print(output)
        gt.extend(target.tolist())
        predicted = prediction2label(output, apply_sigmoid=True)
        # _, predicted = output.max(1)
        pred.extend(predicted.tolist())
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=None)
    print("cohen kappa score: %f" % score)

    sample_weight = sklearn.utils.class_weight.compute_sample_weight(class_weight="balanced", y=gt, indices=None)
    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=sample_weight)
    print("cohen kappa score with sample weight: %f" % score)

    metric_logger.meters['ck'].update(score, n=len(data_loader))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} CK_score {ck.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, ck=metric_logger.ck, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, pred


@torch.no_grad()
def evaluate_tta(data_loader, model, device, use_amp=False):
    transforms = tta.Compose(
        [
           tta.HorizontalFlip(),
#            tta.VerticalFlip(),
#            tta.FiveCrops(224, 224),
#            tta.Rotate90(angles=[0, 180]),
#            tta.Scale(scales=[1, 2, 4]),
#            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )

    model = tta.ClassificationTTAWrapper(model, transforms)

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    gt = []
    pred = []

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)


        #print(output)
        gt.extend(target.tolist())
        _, predicted = output.max(1)
        pred.extend(predicted.tolist())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=None)
    print("cohen kappa score: %f" % score)

    sample_weight = sklearn.utils.class_weight.compute_sample_weight(class_weight="balanced", y=gt, indices=None)
    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=sample_weight)
    print("cohen kappa score with sample weight: %f" % score)

    metric_logger.meters['ck'].update(score, n=len(data_loader))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} CK_score {ck.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, ck=metric_logger.ck, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, pred


@torch.no_grad()
def evaluate_tta_reversed(data_loader, model, device, use_amp=False):
    transforms = tta.Compose(
        [
           tta.HorizontalFlip(),
#            tta.VerticalFlip(),
#            tta.FiveCrops(224, 224),
#            tta.Rotate90(angles=[0, 180]),
#            tta.Scale(scales=[1, 2, 4]),
#            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )

    model = tta.ClassificationTTAWrapper(model, transforms)

    criterion = OrdinalRegressionLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    gt = []
    pred = []

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)


        #print(output)
        gt.extend(target.tolist())
        # _, predicted = output.max(1)
        predicted = prediction2label(output, apply_sigmoid=True)
        pred.extend(predicted.tolist())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=None)
    print("cohen kappa score: %f" % score)

    sample_weight = sklearn.utils.class_weight.compute_sample_weight(class_weight="balanced", y=gt, indices=None)
    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=sample_weight)
    print("cohen kappa score with sample weight: %f" % score)

    metric_logger.meters['ck'].update(score, n=len(data_loader))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} CK_score {ck.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, ck=metric_logger.ck, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, pred



@torch.no_grad()
def evaluate_tta_ensemble(data_loader, models, device, use_amp=False):
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
#            tta.FiveCrops(224, 224),
#            tta.Rotate90(angles=[0, 180]),
#            tta.Scale(scales=[1, 2, 4]),
#            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )

    for i in range(len(models)):
        models[i] = tta.ClassificationTTAWrapper(model=models[i], transforms=transforms)
        models[i].eval()

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    gt = []
    pred = []

    # switch to evaluation mode
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        cnt = 0
        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                for model in models:
                    if cnt == 0:
                        output = softmax(model(images))
                    else:
                        output += softmax(model(images))

                    loss = criterion(output, target)
                    cnt += 1
        else:
            for model in models:
                if cnt == 0:
                    output = softmax(model(images))
                else:
                    output += softmax(model(images))

                loss = criterion(output, target)
                cnt += 1


        #print(output)
        gt.extend(target.tolist())
        _, predicted = output.max(1)
        pred.extend(predicted.tolist())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=None)
    print("cohen kappa score: %f" % score)

    sample_weight = sklearn.utils.class_weight.compute_sample_weight(class_weight="balanced", y=gt, indices=None)
    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=sample_weight)
    print("cohen kappa score with sample weight: %f" % score)

    metric_logger.meters['ck'].update(score, n=len(data_loader))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} CK_score {ck.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, ck=metric_logger.ck, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, pred

@torch.no_grad()
def evaluate_tta_ensemble_reversed(data_loader, models, device, use_amp=False):
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
#            tta.FiveCrops(224, 224),
#            tta.Rotate90(angles=[0, 180]),
#            tta.Scale(scales=[1, 2, 4]),
#            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )

    for i in range(len(models)):
        models[i] = tta.ClassificationTTAWrapper(model=models[i], transforms=transforms)
        models[i].eval()

    criterion = OrdinalRegressionLoss()
    sigmoid = torch.nn.Sigmoid()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    gt = []
    pred = []
    
    # switch to evaluation mode
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        cnt = 0
        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                for model in models:
                    if cnt == 0:
                        output = sigmoid(model(images))
                    else:
                        output += sigmoid(model(images))
                    
                    loss = criterion(output, target)
                    cnt += 1
        else:
            for model in models:
                if cnt == 0:
                    output = sigmoid(model(images))
                else:
                    output += sigmoid(model(images))

                loss = criterion(output, target)
                cnt += 1
        gt.extend(target.tolist())
        predicted = prediction2label(output / len(models), apply_sigmoid=False) 
        # print(predicted)
        pred.extend(predicted.tolist()) 

        # values.append(value)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)


    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=None)
    print("cohen kappa score: %f" % score)

    sample_weight = sklearn.utils.class_weight.compute_sample_weight(class_weight="balanced", y=gt, indices=None)
    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=sample_weight)
    print("cohen kappa score with sample weight: %f" % score)

    metric_logger.meters['ck'].update(score, n=len(data_loader))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} CK_score {ck.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, ck=metric_logger.ck, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, gt, pred #, values1, values2

  
    
def save_tta_ensemble_results(data_loader, dataset, models, device, use_amp=False, is_train=False):
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
        ]
    )

    results = []
    gt = []

    for i in range(len(models)):
        models[i] = tta.ClassificationTTAWrapper(model=models[i], transforms=transforms)
        models[i].eval()
        results.append([])

    sigmoid = torch.nn.Sigmoid()

    metric_logger = utils.MetricLogger(delimiter="  ")

    if is_train:
        header = 'Train:'
    else:
        header = 'Test:'

    cnt = 0
    # switch to evaluation mode
    for batch in metric_logger.log_every(data_loader, 10, header):
        outputs = []
        images = batch[0]
        images = images.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                i = 0
                for model in models:
                    output = sigmoid(model(images))
                    results[i].extend(output.cpu().tolist())
                    del output
                    i += 1

        target = batch[-1]
        target = target.to(device, non_blocking=True)
        gt.extend(target.tolist())

        cnt += 1



    filenames = []
    for f in dataset.imgs:
        a = f[0].split("/")
        filenames.append(a[-1])

    if is_train:
        result_dict = {"model_num": len(models), "features": results, "label": gt, "filename": filenames}

        with open('./data/train_data_final.pkl', 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        result_dict = {"model_num": len(models), "features": results, "label": gt, "filename": filenames}

        with open('./data/test_data_final.pkl', 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return results, gt