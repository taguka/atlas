import os
import time
import cv2
import numpy as np
import pandas as pd
from dataset import HumanDataset, get_tags
from utils import AverageMeter, get_outdir
import torch
import torch.autograd as autograd
import torch.utils.data as data
from models import get_net

class DefaultConfigs(object):
    data = 'train/' # path to dataset
    model = 'resnet101' # Name of model to train (default: "countception"
    multi_label = True # Multi-label target
    tta = 0 # Test/inference time augmentation (oversampling) factor. 0=None (default: 0)
    png = True
    img_size = 512
    batch_size = 8
    seed = 1
    log_interval = 1000 # how many batches to wait before logging training status
    num_processes = 2 # how many training processes to use
    resume = '' # path to restore checkpoint
    no_cuda = False # disables CUDA training
    num_gpu = 1
    num_classes = 28
    channels = 4
    fold = 0
    output = '/content/gdrive/My Drive/output/' # path to output folder
    
config = DefaultConfigs()

def main():

    batch_size = config.batch_size
    img_size = (config.img_size, config.img_size)
    num_classes = config.num_classes
    if config.png:
        img_type = '.png'
    else:
        img_type = '.jpg'

    dataset = HumanDataset(
        config.data,
        train=False,
        multi_label=config.multi_label,
        tags_type='all',
        img_type=img_type,
        img_size=img_size,
        test_aug=config.tta,
    )

    tags = get_tags()
    output_col = ['Id'] + tags
    submission_col = ['Id', 'tags']

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_processes)

    model = get_net(config.model, num_classes=config.num_classes, channels=config.channels)

    if not config.no_cuda:
        if config.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(config.num_gpu))).cuda()
        else:
            model.cuda()

    if config.restore_checkpoint is not None:
        assert os.path.isfile(config.resume), '%s not found' % config.resume
        checkpoint = torch.load(config.resume)
        print('Restoring model with %s architecture...' % checkpoint['arch'])
      
        model.load_state_dict(checkpoint['state_dict'])

        if 'threshold' in checkpoint:
            threshold = checkpoint['threshold']
            threshold = torch.FloatTensor(threshold)
            print('Using thresholds:', threshold)
            if not config.no_cuda:
                threshold = threshold.cuda()
        else:
            threshold = 0.5

        csplit = os.path.normpath(config.resume).split(sep=os.path.sep)
        if len(csplit) > 1:
            exp_name = csplit[-2] + '-' + csplit[-1].split('.')[0]
        else:
            exp_name = ''
        print('Model restored from file: %s' % config.resume)
    else:
        assert False and "No checkpoint specified"

    if config.output:
        output_base = config.output
    else:
        output_base = os.path.join('/content/gdrive/My Drive/','output')
    if not exp_name:
        exp_name = '-'.join([
            config.model,
            str(config.img_size),
            'f'+str(config.fold),
            'png' if config.png else 'jpg'])
    output_dir = get_outdir(output_base, 'predictions', exp_name)

    model.eval()

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    results_raw = []
    results_thr = []
    results_sub = []
    try:
        end = time.time()
        for batch_idx, (input, target, index) in enumerate(loader):
            data_time_m.update(time.time() - end)
            if not config.no_cuda:
                input = input.cuda()
            input_var = autograd.Variable(input, volatile=True)
            output = model(input_var)

            # augmentation reduction
            reduce_factor = loader.dataset.get_aug_factor()
            if reduce_factor > 1:
                output.data = output.data.unfold(0, reduce_factor, reduce_factor).mean(dim=2).squeeze(dim=2)
                index = index[0:index.size(0):reduce_factor]

            # output non-linearity and thresholding
            output = torch.sigmoid(output)
            if isinstance(threshold, torch.FloatTensor) or isinstance(threshold, torch.cuda.FloatTensor):
                threshold_m = torch.unsqueeze(threshold, 0).expand_as(output.data)
                output_thr = (output.data > threshold_m).byte()
            else:
                output_thr = (output.data > threshold).byte()

            # move data to CPU and collect
            output = output.cpu().data.numpy()
            output_thr = output_thr.cpu().numpy()
            index = index.cpu().numpy().flatten()
            for i, o, ot in zip(index, output, output_thr):
                #print(dataset.inputs[i], o, ot)
                image_name = os.path.splitext(os.path.basename(dataset.inputs[i]))[0]
                results_raw.append([image_name] + list(o))
                results_thr.append([image_name] + list(ot))
                results_sub.append([image_name] + [vector_to_tags(ot, tags)])
                # end iterating through batch

            batch_time_m.update(time.time() - end)
            if batch_idx % config.log_interval == 0:
                print('Inference: [{}/{} ({:.0f}%)]  '
                      'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                      '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    batch_idx * len(input), len(loader.sampler),
                    100. * batch_idx / len(loader),
                    batch_time=batch_time_m,
                    rate=input_var.size(0) / batch_time_m.val,
                    rate_avg=input_var.size(0) / batch_time_m.avg,
                    data_time=data_time_m))

            end = time.time()
            #end iterating through dataset
    except KeyboardInterrupt:
        pass
    results_raw_df = pd.DataFrame(results_raw, columns=output_col)
    results_raw_df.to_csv(os.path.join(output_dir, 'results_raw.csv'), index=False)
    results_thr_df = pd.DataFrame(results_thr, columns=output_col)
    results_thr_df.to_csv(os.path.join(output_dir, 'results_thr.csv'), index=False)
    results_sub_df = pd.DataFrame(results_sub, columns=submission_col)
    results_sub_df.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)


def vector_to_tags(v, tags):
    idx = np.nonzero(v)
    t = [tags[i] for i in idx[0]]
    return ' '.join(t)

if __name__ == '__main__':
    main()
