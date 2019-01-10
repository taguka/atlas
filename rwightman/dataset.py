
import cv2
import torch
import torch.utils.data as data
import random
import pandas as pd
import numpy as np
import math
import os
import mytransforms
import utils
import re
from torchvision import transforms

#BASE_PATH = 'C:\\Kaggle\\atlas\\rwightman\\data'
#TRAIN_CSV = 'train.csv'

IMG_EXTENSIONS = [ '.png']
LABEL_ALL = list(map(str,range(28)))

def create_class_weight(labels_dict, mu=0.8):
  total = sum(labels_dict.values())
  keys = labels_dict.keys()
  class_weight = dict()
  class_weight_log = dict()

  for key in keys:
      score = total / float(labels_dict[key])
      score_log = math.log(mu * total / float(labels_dict[key]))
      class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
      class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)
  return class_weight, class_weight_log
"""
train_df = pd.read_csv(os.path.join(BASE_PATH,TRAIN_CSV))
train_df.Target = train_df.Target.map(lambda x: set(x.split()))
count = Counter()
train_df.Target.apply(lambda x: count.update(x))
labels_dict=dict(count)
true_class_weights=create_class_weight(labels_dict)[0]
log_class_weights=create_class_weight(labels_dict)[1]

ALL_WEIGHTS=[true_class_weights[key] for key in sorted(true_class_weights.keys(), 
             key=lambda x:int(x))]
    
ALL_WEIGHTS_L=[log_class_weights[key] for key in sorted(log_class_weights.keys(), 
             key=lambda x:int(x))]
"""   
ALL_WEIGHTS = [3.94, 40.5, 14.02, 32.53, 27.33, 20.21, 50.38, 18.0, 958.15, 
               1128.49, 1813.64, 46.46, 73.81, 94.57, 47.64, 2418.19, 95.82, 
               241.82, 56.3, 34.27, 295.24, 13.45, 63.32, 17.13, 157.71, 6.17, 
               154.82, 4616.55]

ALL_WEIGHTS_L = [1.15, 3.48, 2.42, 3.26, 3.08, 2.78, 3.7, 2.67, 6.64, 6.81, 
                 7.28, 3.62, 4.08, 4.33, 3.64, 7.57, 4.34, 5.27, 3.81, 3.31, 
                 5.46, 2.38, 3.93, 2.62, 4.84, 1.6, 4.82, 8.21]

def get_labels():
    return LABEL_ALL


def get_labels_size():
    return len(get_labels())


def get_class_weights():
    return np.array(ALL_WEIGHTS_L)



def find_inputs(folder, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((base, abs_filename))
    return inputs

def natural_key(string_):
    #See http://www.codinghorror.com/blog/archives/001018.html
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def get_test_aug(factor):
    if not factor or factor == 1:
        return [
            [False, False, False]]
    elif factor == 4:
        # transpose, v-flip, h-flip
        return [
            [False, False, False],
            [False, False, True],
            [False, True, False],
            [True, True, True]]
    elif factor == 8:
        # return list of all combinations of flips and transpose
        return ((1 & np.arange(0, 8)[:, np.newaxis] // 2**np.arange(2, -1, -1)) > 0).tolist()
    else:
        print('Invalid augmentation factor')
        return [
            [False, False, False]]

class HumanDataset(data.Dataset):
    def __init__(
            self,
            input_root,
            target_file='',
            multi_label=True,
            train=True,
            train_fold=False,
            fold=0,
            img_type='.png',
            img_size=(512, 512),
            test_aug=0,
            transform=None):

        assert img_type in ['.png']
        inputs = find_inputs(input_root, types=[img_type])
        if len(inputs) == 0:
            raise (RuntimeError("Found 0 images in : " + input_root))

        if target_file:
            target_df = pd.read_csv(target_file)
            if train or train_fold:
                target_df = target_df[target_df['fold'] != fold]
            else:
                target_df = target_df[target_df['fold'] == fold]
            target_df.drop(['fold'], 1, inplace=True)

            input_dict = dict(inputs)    
            target_df = target_df[target_df.Id.map(lambda x: x in input_dict)]
            target_df['filename'] = target_df.Id.map(lambda x: input_dict[x])
            self.inputs = target_df['Id'].apply(lambda x:os.path.join(input_root,x)).tolist()

            labels = get_labels()
            self.target_array = target_df.as_matrix(columns=labels).astype(np.float32)
            if not multi_label:
                self.target_array = np.argmax(self.target_array, axis=1)

            self.target_array = torch.from_numpy(self.target_array)
        else:
            assert not train
            inputs = sorted(inputs, key=lambda x: natural_key(x[0]))
            self.target_array = None
            self.inputs = [x[1] for x in inputs]

        self.train = train
        if img_type == '.jpg':
            self.dataset_mean = [0.31535792, 0.34446435, 0.30275137]
            self.dataset_std = [0.05338271, 0.04247036, 0.03543708]
        else:
            # For png
            self.dataset_mean = [0.0804419, 0.05262986, 0.05474701, 0.08270896] 
            self.dataset_std = [0.13000701, 0.08796628, 0.1386317, 0.12718021] 

        self.img_size = img_size
        self.img_type = img_type
        if not train:
            self.test_aug = get_test_aug(test_aug)
        else:
            self.test_aug = []
        if transform is None:
            tfs = []
            if img_type == '.jpg':
                tfs.append(mytransforms.ToTensor())
                if self.train:
                    tfs.append(mytransforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01))
                tfs.append(transforms.Normalize(self.dataset_mean, self.dataset_std))
            else:
                tfs.append(mytransforms.ToTensor())
                if self.train:
                    tfs.append(mytransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
                tfs.append(transforms.Normalize(self.dataset_mean, self.dataset_std))
            self.transform = transforms.Compose(tfs)

    def _load_input(self, index):
        colors = ['red','green','blue','yellow']
        path = self.inputs[index]
        if self.img_type == '.jpg':
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            flags = cv2.IMREAD_GRAYSCALE
            img = [cv2.imread((path+'_'+color+'.png'), flags)
            for color in colors]
            return np.stack(img, axis=-1) 

    def _random_crop_and_transform(self, input_img, scale_range=(1.0, 1.0), rot=0.0):
        angle = 0.
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        trans = random.random() < 0.5
        do_rotate = (rot > 0 and random.random() < 0.2) if not hflip and not vflip else False
        h, w = input_img.shape[:2]

        # Favour rotation/scale choices that involve cropping within image bounds
        attempts = 0
        while attempts < 3:
            if do_rotate:
                angle = random.uniform(-rot, rot)
            scale = random.uniform(*scale_range)
            crop_w, crop_h = utils.calc_crop_size(self.img_size[0], self.img_size[1], angle, scale)
            if crop_w <= w and crop_h <= h:
                break
            attempts += 1

        if crop_w > w or crop_h > h:
            # We can still handle crops larger than the source, add a border
            angle = 0.0
            #scale = 1.0
            border_w = crop_w - w
            border_h = crop_h - h
            input_img = cv2.copyMakeBorder(
                input_img,
                border_h//2, border_h - border_h//2,
                border_w//2, border_w - border_w//2,
                cv2.BORDER_REFLECT_101)
            input_img = np.ascontiguousarray(input_img)  # trying to hunt a pytorch/cuda crash, is was this necessary?
            assert input_img.shape[:2] == (crop_h, crop_w)
        else:
            hd = max(0, h - crop_h)
            wd = max(0, w - crop_w)
            ho = random.randint(0, hd) - math.ceil(hd / 2)
            wo = random.randint(0, wd) - math.ceil(wd / 2)
            cx = w // 2 + wo
            cy = h // 2 + ho
            input_img = utils.crop_center(input_img, cx, cy, crop_w, crop_h)

        #print('hflip: %d, vflip: %d, angle: %f, scale: %f' % (hflip, vflip, angle, scale))
        if angle:
            if trans:
                input_img = cv2.transpose(input_img)
            m_translate = np.identity(3)
            if hflip:
                m_translate[0, 0] *= -1
                m_translate[0, 2] = (self.img_size[0] + crop_w) / 2 - 1
            else:
                m_translate[0, 2] = (self.img_size[0] - crop_w) / 2
            if vflip:
                m_translate[1, 1] *= -1
                m_translate[1, 2] = (self.img_size[1] + crop_h) / 2 - 1
            else:
                m_translate[1, 2] = (self.img_size[1] - crop_h) / 2

            if angle or scale != 1.:
                m_rotate = cv2.getRotationMatrix2D((crop_w / 2, crop_h / 2), angle, scale)
                m_final = np.dot(m_translate, np.vstack([m_rotate, [0, 0, 1]]))
            else:
                m_final = m_translate

            input_img = cv2.warpAffine(input_img, m_final[:2, :], self.img_size, borderMode=cv2.BORDER_REFLECT_101)
        else:
            if trans:
                input_img = cv2.transpose(input_img)
            if hflip or vflip:
                if hflip and vflip:
                    c = -1
                else:
                    c = 0 if vflip else 1
                input_img = cv2.flip(input_img, flipCode=c)

            input_img = cv2.resize(input_img, self.img_size,  interpolation=cv2.INTER_LINEAR)

        return input_img

    def _centre_crop_and_transform(self, input_img, scale=1.0, trans=False, vflip=False, hflip=False):
        h, w = input_img.shape[:2]
        cx = w // 2
        cy = h // 2
        crop_w, crop_h = utils.calc_crop_size(self.img_size[0], self.img_size[1], scale=scale)
        input_img = utils.crop_center(input_img, cx, cy, crop_w, crop_h)
        if trans:
            input_img = cv2.transpose(input_img)
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            input_img = cv2.flip(input_img, flipCode=c)
        if scale != 1.0:
            input_img = cv2.resize(input_img, self.img_size, interpolation=cv2.INTER_LINEAR)
        return input_img

    def __getitem__(self, index):
        if not self.train and len(self.test_aug) > 1:
            aug_index = index % len(self.test_aug)
            index = index // len(self.test_aug)
        else:
            aug_index = 0
        input_img = self._load_input(index)
        if self.target_array is not None:
            target_tensor = self.target_array[index]
        else:
            target_tensor = torch.zeros(1)

        h, w = input_img.shape[:2]
        if self.train:
            mid = float(self.img_size[0]) / w
            scale = (mid - .03, mid + .05)

            # size specific overrides
            #if self.img_size[0] == 299:
            #    scale = (1.136, 1.2)  # 299
            #if self.img_size[0] == 256:
            #    scale = (.98, 1.02)  # 256
            #if self.img_size[0] == 237:
            #    scale = (.90, .96)  # 256
            #if self.img_size[0] == 224:
            #    scale = (.86, .90)  # 224

            input_img = self._random_crop_and_transform(input_img, scale_range=scale, rot=5.0)
            input_tensor = self.transform(input_img)
        else:
            scale = float(self.img_size[0]) / w

            # size specific overrides
            #if self.img_size[0] == 299:
            #    scale = 1.168
            if self.img_size[0] == 267:
                scale = 1.05534
            #if self.img_size[0] == 256:
            #    scale = 1.0
            #if self.img_size[0] == 237:
            #    scale = 0.93
            if self.img_size[0] == 224:
                scale = .9

            trans, vflip, hflip = self.test_aug[aug_index] if len(self.test_aug) > 1 else (False, False, False)
            input_img = self._centre_crop_and_transform(
                input_img, scale=scale, trans=trans, vflip=vflip, hflip=hflip)
            input_tensor = self.transform(input_img)

        index_tensor = torch.LongTensor([index])
        return input_tensor, target_tensor, index_tensor

    def __len__(self):
        return len(self.inputs) * len(self.test_aug) if self.test_aug else len(self.inputs)

    def get_aug_factor(self):
        return len(self.test_aug)

    def get_class_weights(self):
        return get_class_weights()

    def get_sample_weights(self):
        class_weights = torch.FloatTensor(self.get_class_weights())
        weighted_samples = []
        for index in range(len(self.inputs)):
            masked_weights = self.target_array[index] * class_weights
            weighted_samples.append(masked_weights.max())
        weighted_samples = torch.DoubleTensor(weighted_samples)
        weighted_samples = weighted_samples / weighted_samples.min()
        return weighted_samples

class WeightedRandomOverSampler(Sampler):
    #Over-samples elements from [0,..,len(weights)-1] factor number of times.
    #Each element is sample at least once, the remaining over-sampling is determined
    #by the weights.
    #Arguments:
    #    weights (list) : a list of weights, not necessary summing up to one
    #    factor (float) : the oversampling factor (>= 1.0)
   

    def __init__(self, weights, factor=2.):
        self.weights = torch.DoubleTensor(weights)
        assert factor >= 1.
        self.num_samples = int(len(self.weights) * factor)

    def __iter__(self):
        base_samples = torch.arange(0, len(self.weights)).long()
        remaining = self.num_samples - len(self.weights)
        over_samples = torch.multinomial(self.weights, remaining, True)
        samples = torch.cat((base_samples, over_samples), dim=0)
        print('num samples', len(samples))
        return (samples[i] for i in torch.randperm(len(samples)))

    def __len__(self):
        return self.num_samples
