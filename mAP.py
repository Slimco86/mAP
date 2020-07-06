import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader 
from src.dataset2 import *
import pickle

class Mapper():
    def __init__(self,model_path,data_path,n_classes=5,nms_threshold=0.5):
        self.n_classes = n_classes
        self.nms_threshold = nms_threshold
        self.threshold = list(np.arange(0,1.1,0.1))
        self.counts = {cls:{'Value':[],'Score':[]} for cls in range(n_classes)}
        #self.counts = {tr:{cls:[] for cls in range(n_classes)} for tr in self.threshold}
        self.model = self.Load_Model(model_path)
        self.dataset = self.Load_Dataset(data_path)
        self.MAP={}
    
    def Load_Model(self,path):
        if not torch.cuda.is_available():
            model = torch.load(path,map_location=torch.device('cpu')).module
        else:
            model = torch.load(path).module
        model.eval()
        print("Model is loaded\n")
        return model

    def Load_Dataset(self,data_path,batch_size=1,num_workers=4):
        params = {'batch_size':batch_size,
                  'num_workers':num_workers,
                  'shuffle':True,
                  'drop_last':True}
        dataset = YoloDataset(root_dir = data_path,class_file='./rovis.names',
                              transform=transforms.Compose([Resizer(common_size = 512),Normalizer()]))
        data_gen = DataLoader(dataset,**params)
        print(f'Data generator is ready.\nTotal amount of samples is {len(data_gen)}.')
        return data_gen

    def IOU(self,bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = np.round(interArea / (boxAArea + np.transpose(boxBArea) - interArea),3)
        return iou
    
    def Check_Img(self,ground_truth,preds):
        true_cls = ground_truth[:,-1] # get list of correct classes
        pred_cls = preds[:,-2] # get list of predicted classes
        pred_conf = preds[:,-1] # get list of prediction confidence
        iou = self.IOU(ground_truth[:,:4],preds[:,:4]) # calculate intersection over union
        ind = np.transpose((iou>self.nms_threshold).nonzero()) # get the boxes indicies that intersect
        sm_ind = np.transpose((iou<self.nms_threshold).nonzero()) # get boxes indices that intersect with iou<threshold

        for pair in sm_ind:
            if true_cls[pair[0]]==int(pred_cls[pair[1]]):
                self.counts[true_cls[pair[0]]]['Value'].append('FP') # False positive if class is correct but IOU<threshold
                self.counts[true_cls[pair[0]]]['Score'].append(pred_conf[pair[1]])
        for pair in ind:    # for intersection pairs
            if true_cls[pair[0]]==int(pred_cls[pair[1]]):
                self.counts[true_cls[pair[0]]]['Value'].append('TP') # True positive if classes match
                self.counts[true_cls[pair[0]]]['Score'].append(pred_conf[pair[1]])
            elif true_cls[pair[0]]!=int(pred_cls[pair[1]]):
                self.counts[true_cls[pair[0]]]['Value'].append('FN')  # False negative if classes do not match
                self.counts[true_cls[pair[0]]]['Score'].append(pred_conf[pair[1]])
        
        pred_boxes = ind[:,0]
        unpr_boxes = np.setdiff1d([i for i in range(ground_truth.shape[0])], pred_boxes) # Get boxes indices for which no bbox was predicted
        for box in unpr_boxes:
            self.counts[true_cls[box]]['Value'].append('FN') # set false negative for missed boxes
            self.counts[true_cls[box]]['Score'].append(pred_conf[box])
        return 

    def Check_Dataset(self):
        progress_bar = tqdm(self.dataset)  # pass the dataset               
        for iter,sample in enumerate(progress_bar):   # iterate dataset
            ground_truth = sample['annot'].numpy().reshape([-1,5]) # get ground truth
            if torch.cuda.is_available():   #  inference on GPU
                score,clas,box = self.model(sample['img'].cuda())
                box,clas,score = box.detach().cpu().numpy(),clas.detach().cpu().numpy(),score.detach().cpu().numpy() # Convert results to numpy
            else:  # inference on CPU
                score,clas,box = self.model(sample['img'])
                box,clas,score = box.detach().numpy(),clas.detach().numpy(),score.detach().numpy() # Convert results to numpy
            preds = np.concatenate((box,clas[:,np.newaxis],score[:,np.newaxis]),axis=1)     # concatenate results into predictions      
            #for threshold in self.threshold:  #  fore each score threshold calculate recal-precision
                #trunc_pred = preds[preds[:,-1]>=threshold]  
            self.Check_Img(ground_truth,preds)

        with open('results.pkl','wb') as f:
            pickle.dump(self.counts,f)


    def Get_Metrics(self,cls,threshold):
        TP = self.counts[threshold][cls].count('TP')
        FP = self.counts[threshold][cls].count('FP')
        FN = self.counts[threshold][cls].count('FN')
        if TP == 0:
            return (0,0,0)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        accuracy = TP/len(self.counts[cls])
        return (precision, recall, accuracy)

    def Calc_mAP(self):
        assert len(self.counts[0])>0,print('First analize the dataset')
        for threshold in self.threshold:
            if threshold not in self.MAP.keys():
                self.MAP[threshold]={}
            for cls in self.n_classes:
                p,r = self.AP(cls,threshold)
                self.MAP[threshold][cls]=(p,r)


    def AP(self,cls,threshold,graph=False):
        TP_tot = self.counts[threshold][cls].count('TP')
        TP = 0
        FP = 0
        FN = 0
        precision = [1]
        recall = [0]
        for q,pred in enumerate(self.counts[threshold][cls]):
            
            if pred == 'TP':
                TP+=1
            recall.append(TP/TP_tot)
            precision.append(TP/(q+1))

        if graph:
            print(recall,precision)
            plt.plot(recall,precision)
            plt.scatter(recall,precision)
            plt.xlim([0,1.2])
            plt.ylim([0,1.2])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Class: {cls}')       

            plt.show()
        return (precision,recall)
            



if __name__ == '__main__':
    model_path = '/media/linfile1/users/ivoloshenko/Documents/Repositories/efficientdet-torch/trained_models/b0/RoVis_efficientdet_yolo.pth'
    data_path = '/media/linfile1/users/ivoloshenko/Documents/RoVis/Datasets/Object_detection/YOLO_DATA/with_art_data/test2'
 
    maper = Mapper(model_path,data_path,n_classes=5)
    maper.Check_Dataset()
    print(maper.Get_Metrics(1,0.5))
    maper.AP(1,0.5)
