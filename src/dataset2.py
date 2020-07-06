import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import cv2



#"/media/linfile1/users/ivoloshenko/Documents/Repositories/efficientdet/rovis.names"
class YoloDataset(Dataset):
    def __init__(self,root_dir,class_file="/media/linfile1/users/ivoloshenko/Documents/Repositories/efficientdet-torch/rovis.names",transform=None):
        #super().__init__(YoloDataset,self)

        self.root_dir = root_dir
        self.transform = transform
        self.class_file = class_file
        self.classes = self.load_classes()
        self.img_ids = self.get_img_ids()
    
    def load_classes(self):
        classes=[]
        with open(self.class_file,'r') as f:
            for line in f:
                classes.append(line.split('\n')[0])
        return classes

    def get_img_ids(self):
        img_ids = []
        for file in os.listdir(self.root_dir):
            if file.endswith('.jpg'):
                img_ids.append(file.split('.jpg')[0])
        return img_ids
    
    def __len__(self):
        return len(self.img_ids)
    

    def __getitem__(self, idx):
    
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        path = os.path.join(self.root_dir, self.img_ids[image_index]+'.jpg')
        try:
            img = Image.open(path)
        except PermissionError:
            return Image.fromarray(np.zeros((500,500,3)).astype(np.uint8))
        return img

    def load_annotations(self, image_index):
        path = os.path.join(self.root_dir, self.img_ids[image_index]+'_coco.txt')
        annotations = np.zeros((0,5))
        with open(path,'r') as f:
            for line in f:
                if line in ['\n','r\n']:
                    continue
                vals = [float(s) for s in line.split('\t') if (s!='\t' and s!='\n')]
                #vals[-1] = int(vals[-1])
                #print("{} : {}".format(self.img_ids[image_index],np.asarray(vals).shape))
                annotations = np.append(annotations,np.asarray(vals).reshape((1,5)),axis=0)
        return annotations

    def num_classes(self):
        return len(self.classes)


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    return {'img': torch.stack(imgs), 'annot': annot_padded}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,common_size=512):
        self.common_size=common_size

    def __call__(self, sample ):
        image, annots = sample['img'], sample['annot']

        width, height = image.size
        if height > width:
            scale = self.common_size / height
            resized_height = self.common_size
            resized_width = int(width * scale)
            annots[:,1:5:2] *= resized_height
            annots[:,0:4:2] *= resized_width
        else:
            scale = self.common_size / width
            resized_height = int(height * scale)
            resized_width = self.common_size
            annots[:,1:5:2] *= resized_height
            annots[:,0:4:2] *= resized_width            
        rs = transforms.Resize((resized_height, resized_width))
        image = rs(image)
        new_image = np.zeros((self.common_size, self.common_size, 3))
        try:
            new_image[0:resized_height, 0:resized_width,:] = np.array(image)
        except ValueError:
            new_image[0:resized_height, 0:resized_width,:] = np.repeat(np.array(image)[:, :, np.newaxis],3,axis=2)
        return {'img': transforms.ToTensor()(new_image.astype(np.uint8)), 'annot': torch.from_numpy(annots.astype(int))}



class Normalizer(object):
    
    def __init__(self):
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])
        self.norm = transforms.Normalize(mean=self.mean,std=self.std)
    def __call__(self, sample):
        image, annots, = sample['img'], sample['annot']
        image = self.norm(image)
        return {'img': image, 'annot': annots}



class Color_Jitter(object):
    def __init__(self,s=1,probability = 0.8):
        self.s = s
        self.prob = probability
    def __call__(self,sample):
        image, annots = sample['img'], sample['annot']
        image = transforms.ToPILImage()(image)
        color_jitter = transforms.ColorJitter(0.8*self.s, 0.8*self.s, 0.8*self.s, 0.2*self.s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=self.prob)
        out_image = rnd_color_jitter(image)
        out_image = transforms.ToTensor()(out_image)
        return {'img': out_image, 'annot': annots}

class RandomGaussian(object):
    def __init__(self):
        self.p = 0.5
        
    def __call__(self,sample):
        toss = np.random.rand()
        image,annots = sample['img'],sample['annot']
        if toss>=self.p:
            noise = torch.normal(0.,0.01,image.size())
            return {'img': image+noise, 'annot': annots}
        else:
            return sample


class Rotator(object):
    def __init__(self):
        self.angle = np.random.uniform(0,180)
        self.chance = 0.5
    def __call__(self,sample):

        image,annots = sample['img'],sample['annot']
        image = transforms.ToPILImage()(image)
        rows,cols = image.size
        rotor = transforms.RandomRotation(self.angle)
        image = rotor(image)
        if annots.shape[0]!=0:
            annots = self.rotate_bbox(annots,cols,rows)
        return {'img': transforms.ToTensor()(image), 'annot': annots}

    def rotate_bbox(self,bbox,cols,rows):
        labels = bbox[:,-1:].numpy()
        x_min = bbox[:,0].numpy()
        y_min = bbox[:,1].numpy()
        x_max = bbox[:,2].numpy()
        y_max = bbox[:,3].numpy()
        scale = cols / float(rows)
        x = np.array([x_min, x_max, x_max, x_min]) - cols//2
        y = np.array([y_min, y_min, y_max, y_max]) - rows//2
        angle = np.deg2rad(self.angle)
        x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
        y_t = (-np.sin(angle) * x * scale + np.cos(angle) * y) / scale
        x_t = x_t + cols//2
        y_t = y_t + rows//2
        x_min, x_max = np.min(x_t,axis=0), np.max(x_t,axis=0)
        y_min, y_max = np.min(y_t,axis=0), np.max(y_t,axis=0)
        out = np.stack((x_min,y_min,x_max,y_max),axis=1)
        #print('Out:\n{}'.format(out))
        #print('Labels:\n{}'.format(labels))
        out = np.concatenate((out,labels),axis=1)
        return torch.from_numpy(out.astype(int))