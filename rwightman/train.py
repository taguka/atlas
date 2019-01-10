import csv
import glob
import os
import shutil
import argparse
import time
import numpy as np
from collections import OrderedDict
from datetime import datetime
from sklearn.metrics import fbeta_score

from dataset import HumanDataset, get_tags_size
from models import get_net
from opt.yellowfin import YFOptimizer
from opt.lr_scheduler import ReduceLROnPlateau
from utils import AverageMeter, get_outdir

import torch
import torch.autograd as autograd
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

parser = argparse.ArgumentParser(description='user')
parser.add_argument('--user', metavar='USER',
                    help='colab user')
args = parser.parse_args()
    
class DefaultConfigs(object):
    data = 'train/' # path to dataset
    model = 'resnet101' # Name of model to train (default: "countception"
    opt = 'adam' # Optimizer
    loss = 'mlsm' # Loss function
    multi_label = True # Multi-label target
    tta = 0 # Test/inference time augmentation (oversampling) factor. 0=None (default: 0)
    if args.user =='gpu.colab':
        fold = 0
    elif args.user == 'aguka136':
        fold = 1
    elif args.user == 'squirrel136':
        fold = 2 
    elif args.user == 'tegavrylenko':
        fold = 3  # Train/valid fold
    img_size = 512 # Image patch size 
    batch_size = 8
    test_batch_size = 4
    epochs = 200
    decay_epochs = 15 # epoch interval to decay LR
    ft_epochs = 0 # Number of finetuning epochs (final layer only)
    ft_opt = 'sgd'
    ft_lr = 0.0001 # Finetune learning rates
    drop = 0.1 # Dropout rate
    lr = 0.001 # learning rate
    momentum = 0.9 # SGD momentum
    weight_decay = 0.0001 # weight decay
    if args.user =='gpu.colab':
        seed = 1
    elif args.user == 'aguka136':
        seed = 10
    elif args.user == 'squirrel136':
        seed = 20 
    elif args.user == 'tegavrylenko':
        seed = 30  
    log_interval = 1000 # how many batches to wait before logging training status
    num_processes = 1 # how many training processes to use
    no_cuda = False # disables CUDA training
    no_tb = False # disables tensorboard
    tbh = '127.0.0.1:8009' # Tensorboard (Crayon) host
    num_gpu = 1 # Number of GPUS to use
    checkpoint_path = '/content/gdrive/My Drive/atlas/'
    if args.user =='gpu.colab':
<<<<<<< HEAD
        resume = os.path.join(checkpoint_path, 'checkpoint-18.pth.tar')
    elif args.user == 'aguka136':
        resume = os.path.join(checkpoint_path, 'checkpoint-22.pth.tar')
    elif args.user == 'squirrel136':
        resume = os.path.join(checkpoint_path, 'checkpoint-28.pth.tar')
    elif args.user == 'tegavrylenko':
        resume == os.path.join(checkpoint_path, 'checkpoint-27.pth.tar')  # path to latest checkpoint (default: none)
=======
        resume = os.path.join(checkpoint_path, 'checkpoint-14.pth.tar')
    elif args.user == 'aguka136':
        resume = os.path.join(checkpoint_path, 'checkpoint-17.pth.tar')
    elif args.user == 'squirrel136':
        resume = os.path.join(checkpoint_path, 'checkpoint-23.pth.tar')
    elif args.user == 'tegavrylenko':
        resume =  resume = os.path.join(checkpoint_path, 'checkpoint-33.pth.tar')  # path to latest checkpoint (default: none)
>>>>>>> 6d1f075f8fb0143e5cbe33097a4bda63230d56dd
    print_freq = 200 # print frequency 
    save_batches = False # save images of batch inputs and targets every log interval for debugging/verification
    output = '/gdrive/My Drive/atlas/output/' # path to output folder (default: none, current dir)
    class_weights = False # Use class weights for specified labels as loss penalty
    channels=4
    img_type='.png'

def main():
    config = DefaultConfigs()
    train_input_root = os.path.join(config.data)
    train_labels_file = 'labels.csv'

    if config.output:
        if not os.path.exists(config.output):
            os.makedirs(config.output)
        output_base = config.output
    else:
        if not os.path.exists(config.output):
            os.makedirs(config.output)
        output_base = config.output

    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        config.model,
        str(config.img_size),
        'f'+str(config.fold)])
    mask_exp_name = '-'.join([
        config.model,
        str(config.img_size),
        'f'+str(config.fold)])
    mask_exp_name=glob.glob(os.path.join(output_base,
                                                'train','*' + mask_exp_name))
    if config.resume and mask_exp_name:
        output_dir=mask_exp_name
    else:
        output_dir = get_outdir(output_base, 'train', exp_name)

    batch_size = config.batch_size
    test_batch_size = config.test_batch_size
    num_epochs = config.epochs
    img_type = config.image_type
    img_size = (config.img_size, config.img_size)
    num_classes = get_tags_size(config.labels)

    torch.manual_seed(config.seed)

    dataset_train = HumanDataset(
        train_input_root,
        train_labels_file,
        train=True,
        multi_label=config.multi_label,
        img_type=img_type,
        img_size=img_size,
        fold=config.fold,
    )

    #sampler = WeightedRandomOverSampler(dataset_train.get_sample_weights())

    loader_train = data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        #sampler=sampler,
        num_workers=config.num_processes
    )

    dataset_eval = HumanDataset(
        train_input_root,
        train_labels_file,
        train=False,
        multi_label=config.multi_label,
        img_type=img_type,
        img_size=img_size,
        test_aug=config.tta,
        fold=config.fold,
    )

    loader_eval = data.DataLoader(
        dataset_eval,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=config.num_processes
    )

#    model = model_factory.create_model(
#        config.model,
#        pretrained=True,
#        num_classes=num_classes,
#        drop_rate=config.drop,
#        global_pool=config.gp)

    model=get_net(config.model, num_classes, config.drop, config.channels)

    if not config.no_cuda:
        if config.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(config.num_gpu))).cuda()
        else:
            model.cuda()

    if config.opt.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.opt.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.opt.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(), lr=config.lr, alpha=0.9, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.opt.lower() == 'yellowfin':
        optimizer = YFOptimizer(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay, clip_thresh=2)
    else:
        assert False and "Invalid optimizer"

    if not config.decay_epochs:
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=8)
    else:
        lr_scheduler = None

    if config.class_weights:
        class_weights = torch.from_numpy(dataset_train.get_class_weights()).float()
        class_weights_norm = class_weights / class_weights.sum()
        if not config.no_cuda:
            class_weights = class_weights.cuda()
            class_weights_norm = class_weights_norm.cuda()
    else:
        class_weights = None
        class_weights_norm = None

    if config.loss.lower() == 'nll':
        #assert not args.multi_label and 'Cannot use crossentropy with multi-label target.'
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif config.loss.lower() == 'mlsm':
        assert config.multi_label
        loss_fn = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
    else:
        assert config and "Invalid loss function"

    if not config.no_cuda:
        loss_fn = loss_fn.cuda()

    # optionally resume from a checkpoint
    start_epoch = 1
    if config.resume:
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            config.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(config.resume, checkpoint['epoch']))
            start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))
            exit(-1)

    use_tensorboard = not config.no_tb and CrayonClient is not None
    if use_tensorboard:
        hostname = '127.0.0.1'
        port = 8889
        host_port = config.tbh.split(':')[:2]
        if len(host_port) == 1:
            hostname = host_port[0]
        elif len(host_port) >= 2:
            hostname, port = host_port[:2]
        try:
            cc = CrayonClient(hostname=hostname, port=port)
            try:
                cc.remove_experiment(exp_name)
            except ValueError:
                pass
            exp = cc.create_experiment(exp_name)
        except Exception as e:
            exp = None
            print("Error (%s) connecting to Tensoboard/Crayon server. Giving up..." % str(e))
    else:
        exp = None

    # Optional fine-tune of only the final classifier weights for specified number of epochs (or part of)
    if not config.resume and config.ft_epochs > 0.:
        if config.opt.lower() == 'adam':
            finetune_optimizer = optim.Adam(
                model.get_fc().parameters(), lr=config.ft_lr, weight_decay=config.weight_decay)
        else:
            finetune_optimizer = optim.SGD(
                model.get_fc().parameters(), lr=config.ft_lr, momentum=config.momentum, weight_decay=config.weight_decay)

        finetune_epochs_int = int(np.ceil(config.ft_epochs))
        finetune_final_batches = int(np.ceil((1 - (finetune_epochs_int - config.ft_epochs)) * len(loader_train)))
        print(finetune_epochs_int, finetune_final_batches)
        for fepoch in range(1, finetune_epochs_int + 1):
            if fepoch == finetune_epochs_int and finetune_final_batches:
                batch_limit = finetune_final_batches
            else:
                batch_limit = 0
            train_epoch(
                fepoch, model, loader_train, finetune_optimizer, loss_fn, config,
                class_weights_norm, output_dir, batch_limit=batch_limit)
            step = fepoch * len(loader_train)
            score, _ = validate(step, model, loader_eval, loss_fn, config, 0.3, output_dir)

    score_metric = 'f2'
    best_loss = None
    best_f2 = None
    threshold = 0.2
    try:
        for epoch in range(start_epoch, num_epochs + 1):
            if config.decay_epochs:
                adjust_learning_rate(optimizer, epoch, initial_lr=config.lr, decay_epochs=config.decay_epochs)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, loss_fn, config, class_weights_norm, output_dir, exp=exp)

            step = epoch * len(loader_train)
            eval_metrics, latest_threshold = validate(
                step, model, loader_eval, loss_fn, config, threshold, output_dir, exp=exp)

            if lr_scheduler is not None:
                lr_scheduler.step(eval_metrics['eval_loss'])

            rowd = OrderedDict(epoch=epoch)
            rowd.update(train_metrics)
            rowd.update(eval_metrics)
            with open(os.path.join(output_dir, 'summary.csv'), mode='a') as cf:
                dw = csv.DictWriter(cf, fieldnames=rowd.keys())
                if best_loss is None:  # first iteration (epoch == 1 can't be used)
                    dw.writeheader()
                dw.writerow(rowd)

            best = False
            if best_loss is None or eval_metrics['eval_loss'] < best_loss[1]:
                best_loss = (epoch, eval_metrics['eval_loss'])
                if score_metric == 'loss':
                    best = True
            if best_f2 is None or eval_metrics['eval_f2'] > best_f2[1]:
                best_f2 = (epoch, eval_metrics['eval_f2'])
                if score_metric == 'f2':
                    best = True

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config.model,
                'state_dict':  model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'threshold': latest_threshold,
                'config': config
                },
                is_best=best,
                filename=os.path.join(config.checkpoint_path,'checkpoint-%d.pth.tar' % epoch),
                output_dir=output_dir)

    except KeyboardInterrupt:
        pass
    print('*** Best loss: {0} (epoch {1})'.format(best_loss[1], best_loss[0]))
    print('*** Best f2: {0} (epoch {1})'.format(best_f2[1], best_f2[0]))


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, config,
        class_weights=None, output_dir='', exp=None, batch_limit=0):

    epoch_step = (epoch - 1) * len(loader)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (input, target, index) in enumerate(loader):
        step = epoch_step + batch_idx
        data_time_m.update(time.time() - end)
        if not config.no_cuda:
            input, target = input.cuda(), target.cuda()
        input_var = autograd.Variable(input)

        if config.multi_label and config.loss == 'nll':
            # if multi-label AND nll set, train network by sampling an index using class weights
            if class_weights is not None:
                target_weights = target * torch.unsqueeze(class_weights, 0).expand_as(target)
                ss = target_weights.sum(dim=1, keepdim=True).expand_as(target_weights)
                target_weights = target_weights.div(ss)
            else:
                target_weights = target
            target_var = autograd.Variable(torch.multinomial(target_weights, 1).squeeze().long())
        else:
            target_var = autograd.Variable(target)

        output = model(input_var)
        loss = loss_fn(output, target_var)
        losses_m.update(loss.data.item(), input_var.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time_m.update(time.time() - end)
        if batch_idx % config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  '
                  'Loss: {loss.val:.6f} ({loss.avg:.4f})  '
                  'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                  '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch,
                batch_idx * len(input), len(loader.sampler),
                100. * batch_idx / len(loader),
                loss=losses_m,
                batch_time=batch_time_m,
                rate=input_var.size(0) / batch_time_m.val,
                rate_avg=input_var.size(0) / batch_time_m.avg,
                data_time=data_time_m))

            if exp is not None:
                exp.add_scalar_value('loss_train', losses_m.val, step=step)
                exp.add_scalar_value('learning_rate', optimizer.param_groups[0]['lr'], step=step)

            if config.save_batches:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)

        end = time.time()

        if batch_limit and batch_idx >= batch_limit:
            break

    return OrderedDict([('train_loss', losses_m.avg)])


def validate(step, model, loader, loss_fn, config, threshold, output_dir='', exp=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    acc_m = AverageMeter()
    f2_m = AverageMeter()

    model.eval()

    end = time.time()
    output_list = []
    target_list = []
    for i, (input, target, _) in enumerate(loader):
        if not config.no_cuda:
            input, target = input.cuda(), target.cuda()
        if config.multi_label and config.loss == 'nll':
            # pick one of the labels for validation loss, should we randomize like in train?
            target_var = autograd.Variable(target.max(dim=1)[1].squeeze(), volatile=True)
        else:
            target_var = autograd.Variable(target, volatile=True)
        input_var = autograd.Variable(input, volatile=True)

        # compute output
        output = model(input_var)

        # augmentation reduction
        reduce_factor = loader.dataset.get_aug_factor()
        if reduce_factor > 1:
            output.data = output.data.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
            target_var.data = target_var.data[0:target_var.size(0):reduce_factor]

        # calc loss
        loss = loss_fn(output, target_var)
        losses_m.update(loss.data.item(), input.size(0))

        # output non-linearities and metrics
        if config.multi_label:
            if config.loss == 'nll':
                output = F.softmax(output)
            else:
                output = torch.sigmoid(output)
            a, p, _, f2 = scores(output.data, target_var.data, threshold)
            acc_m.update(a, output.size(0))
            prec1_m.update(p, output.size(0))
            f2_m.update(f2, output.size(0))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
            prec1_m.update(prec1[0], output.size(0))
            prec5_m.update(prec5[0], output.size(0))

        # copy to CPU and collect
        target_list.append(target.cpu().numpy())
        output_list.append(output.data.cpu().numpy())

        batch_time_m.update(time.time() - end)
        end = time.time()
        if i % config.print_freq == 0:
            if config.multi_label:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Acc {acc.val:.4f} ({acc.avg:.4f})  '
                      'Prec {prec.val:.4f} ({prec.avg:.4f})  '
                      'F2 {f2.val:.4f} ({f2.avg:.4f})  '.format(
                    i, len(loader),
                    batch_time=batch_time_m, loss=losses_m,
                    acc=acc_m, prec=prec1_m, f2=f2_m))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Prec@1 {top1.val:.4f} ({top1.avg:.4f})  '
                      'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                    i, len(loader),
                    batch_time=batch_time_m, loss=losses_m,
                    top1=prec1_m, top5=prec5_m))

            if config.save_batches:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'validate-batch-%d.jpg' % i),
                    padding=0,
                    normalize=True)

    output_total = np.concatenate(output_list, axis=0)
    target_total = np.concatenate(target_list, axis=0)
    if config.multi_label:
        new_threshold, f2 = optimise_f2_thresholds(target_total, output_total)
        metrics = [('eval_loss', losses_m.avg), ('eval_f2', f2)]
    else:
        f2 = f2_score(output_total, target_total, threshold=0.5)
        new_threshold = []
        metrics = [('eval_loss', losses_m.avg), ('eval_f2', f2), ('eval_prec1', prec1_m.avg)]
    print(f2, new_threshold)

    if exp is not None:
        exp.add_scalar_value('loss_eval', losses_m.avg, step=step)
        exp.add_scalar_value('prec@1_eval', prec1_m.avg, step=step)
        exp.add_scalar_value('f2_eval', f2, step=step)

    return OrderedDict(metrics), new_threshold


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs=30):
    #Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    if isinstance(optimizer, YFOptimizer):
        return
    lr = initial_lr * (0.1 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', output_dir=''):
    save_path = os.path.join(output_dir, filename)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(output_dir, 'model_best.pth.tar'))


def accuracy(output, target, topk=(1,)):
    #Computes the precision@k for the specified values of k
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def scores(output, target, threshold=0.5):
    # Count true positives, true negatives, false positives and false negatives.
    outputr = (output > threshold).long()
    target = target.long()
    a_sum = 0.0
    p_sum = 0.0
    r_sum = 0.0
    f2_sum = 0.0

    def _safe_size(t, n=0):
        if n < len(t.size()):
            return t.size(n)
        else:
            return 0

    count = 0
    for o, t in zip(outputr, target):
        tp = _safe_size(torch.nonzero(o * t))
        tn = _safe_size(torch.nonzero((o - 1) * (t - 1)))
        fp = _safe_size(torch.nonzero(o * (t - 1)))
        fn = _safe_size(torch.nonzero((o - 1) * t))
        a = (tp + tn) / (tp + fp + fn + tn)
        if tp == 0 and fp == 0 and fn == 0:
            p = 1.0
            r = 1.0
            f2 = 1.0
        elif tp == 0 and (fp > 0 or fn > 0):
            p = 0.0
            r = 0.0
            f2 = 0.0
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f2 = (5 * p * r) / (4 * p + r)
        a_sum += a
        p_sum += p
        r_sum += r
        f2_sum += f2
        count += 1
    accuracy = a_sum / count
    precision = p_sum / count
    recall = r_sum / count
    fmeasure = f2_sum / count
    return accuracy, precision, recall, fmeasure


def f2_score(output, target, threshold):
    output = (output > threshold)
    return fbeta_score(target, output, beta=2, average='samples')


def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    # Find optimal threshold values for f2 score. Thanks Anokas
    #https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
    
    size = y.shape[1]

    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(size):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = [0.2] * size
    for i in range(size):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)

    return x, best_score

if __name__ == '__main__':
    main()
