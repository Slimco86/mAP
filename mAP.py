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
        #self.model = self.Load_Model(model_path)
        #self.dataset = self.Load_Dataset(data_path)
        self.MAP={}
        self.classes = ['Bad Walls','Dust','Tee','Spiral Weld','Girth Weld']
    
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
    
    def Load_Predictions(self,path):
        with open(path,'rb') as f:
           self.counts=pickle.load(f)

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
            self.Check_Img(ground_truth,preds)
        with open('results.pkl','wb') as f:
            pickle.dump(self.counts,f)

    def Sort_Predictions(self):
        """ Sort the predictions with ascending confidence score"""
        print('Sorting predictions')
        for cls in self.counts.keys():
            values = self.counts[cls]['Value']
            scores = self.counts[cls]['Score']
            ziped = zip(scores,values)
            sorted_pairs = sorted(ziped)
            tuples = zip(*sorted_pairs)

            sc, val = [list(tuple) for tuple in tuples]
            self.counts[cls]['Value'] = list(reversed(val))
            self.counts[cls]['Score'] = list(reversed(sc))



    def Get_Metrics(self):
        """Get global precission, recall, accuracy scores"""
        print('Class:\t\tPrecision:\t\tRecall:\t\tAccuracy"')
        for cls in self.counts.keys():
            TP = self.counts[cls]['Value'].count('TP')
            FP = self.counts[cls]['Value'].count('FP')
            FN = self.counts[cls]['Value'].count('FN')
            if TP == 0:
                return (0,0,0)
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            accuracy = TP/(TP+FN+FP)
            
            print(f'{self.classes[cls]}\t\t{precision}\t\t{recall}\t\t{accuracy}')
            

    def Calc_mAP(self):
        self.PR = {cls:{'P':[],'R':[]} for cls in self.counts.keys()}
        ap = {cls:None for cls in self.counts.keys()}
        for cls in self.counts.keys():
            query = 1
            TP,FP,FN = 0,0,0
            TP_tot = self.counts[cls]['Value'].count('TP')+self.counts[cls]['Value'].count('FN')
            for val,sc in tqdm(zip(self.counts[cls]['Value'],self.counts[cls]['Score'])):
                if val == 'TP':
                    TP+=1
                elif val == 'FN':
                    FN+=1
                elif val == 'FP':
                    FP+=1
                pr = TP/query
                rec = TP/TP_tot
                self.PR[cls]['P'].append(pr)
                self.PR[cls]['R'].append(rec)
                query+=1

            # Smoothing PR curve
            old_ind = 0
            new_p = []
            while old_ind<len(self.PR[cls]['P']):
                mval = max(self.PR[cls]['P'][old_ind:])
                new_ind = self.PR[cls]['P'][old_ind:].index(mval)
                for _ in range(new_ind+1):
                    new_p.append(mval)
                old_ind+=new_ind+1
            self.PR[cls]['P'] = new_p
        
            diff = [self.PR[cls]['R'][i]-self.PR[cls]['R'][i-1] for i in range(1,len(self.PR[cls]['R']))]
            ap[cls] = np.sum(np.array(self.PR[cls]['P'][1:])*np.array(diff))
        print(f'mAP score: {round(sum([i for i in ap.values()])/self.n_classes,2)}')

    def Display_AP(self):
        
        colors = ['r','orange','g','c','b']
        markers = ['o','v','*','1','s']

        for cls in self.PR.keys():
            lab = False
            pr = self.PR[cls]['P']
            rec = self.PR[cls]['R']
            if not lab:
                plt.scatter(rec,pr,c=colors[cls],marker=markers[cls],label=self.classes[cls])
                lab = True
            else:
                plt.scatter(rec,pr,c=colors[cls],marker=markers[cls])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.tick_params(direction='in', length=6, width=3, colors='r',
                       grid_color='r', grid_alpha=0.2)
        plt.title('ROC curve')
        plt.xticks(np.arange(0,1.1,0.1))
        plt.yticks(np.arange(0,1.1,0.1))
        plt.legend()
        plt.show()
            

    def Summary(self):
        for cls in self.counts.keys():
            print(f'{self.classes[cls]} total number of positive samples: {self.counts[cls]["Value"].count("TP")+self.counts[cls]["Value"].count("FN")}')

if __name__ == '__main__':
    model_path = '/media/linfile1/users/ivoloshenko/Documents/Repositories/efficientdet-torch/trained_models/b0/RoVis_efficientdet_yolo.pth'
    data_path = '/media/linfile1/users/ivoloshenko/Documents/RoVis/Datasets/Object_detection/YOLO_DATA/with_art_data/test2'
    maper = Mapper(model_path,data_path,n_classes=5)
    maper.Load_Predictions('./results.pkl')
    maper.Sort_Predictions()
    maper.Summary()
    maper.Get_Metrics()
    maper.Calc_mAP()
    maper.Display_AP()

