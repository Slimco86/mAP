import numpy as np
import cv2
import os
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader 

class Mapper():
    def __init__(self,model_path,data_path,true_labels,pred_labels,n_classes=5):
        self.threshold = list(np.arange(0,1,0.1))
        #self.counts = {cls:[] for cls in range(n_classes)}
        self.counts = {tr:{cls:[] for cls in range(n_classes)} for tr in self.threshold}
        self.dataset = self.Load_Dataset(data_path)
        self.model = Load_Model(model_path)
    
    def Load_Model(self,path):
        model = torch.load(path)
        model.train(False)
        if torch.cuda.is_available():
            model.cuda()
        return model

    def Load_Dataset(self,data_path,batch_size=4,num_workers=4):
        params = {'batch_size':batch_size,
                  'num_workers':num_workers,
                  'shuffle':True,
                  'drop_last':True}
        dataset = Dataset(root_dir = data_path)
        data_gen = DataLoader(dataset,params)
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
    
    def Check_Img(self,ground_truth,preds,threshold):
        true_cls = ground_truth[:,-1] # get list of correct classes
        pred_cls = preds[:,-2] # get list of predicted classes
        pred_conf = preds[:,-1] # get list of prediction confidence
        iou = self.IOU(ground_truth[:,:4],preds[:,:4]) # calculate intersection over union
        ind = np.transpose((iou>self.threshold).nonzero()) # get the boxes indicies that intersect

        for pair in ind:    # for intersection pairs
            if true_cls[pair[0]]==int(pred_cls[pair[1]]):
                self.counts[threshold][true_cls[pair[0]]].append('TP') # True positive if classes match
            elif true_cls[pair[0]]!=int(pred_cls[pair[1]]):
                self.counts[threshold][true_cls[pair[0]]].append('FN')  # False negative if classes do not match
        
        pred_boxes = ind[:,0]
        unpr_boxes = np.setdiff1d([i for i in range(ground_truth.shape[0])], pred_boxes) # Get boxes indices for which no bbox was predicted
        for box in unpr_boxes:
            self.counts[threshold][true_cls[box]].append('FP') # set false positive for missed boxes
        return 

    def Check_Dataset(self):
        
        progress_bar = tqdm(self.dataset)
        for iter,sample in progress_bar:
            if torch.cuda.is_available():
                preds = self.model(sample['img'].cuda())
            else:
                preds = self.model(sample['img'])
            for treshold in self.threshold:
                trunc_pred = preds[preds[:,-1]>=threshold]
                ground_truth = sample['annot']    
                self.Check_Img(ground_truth,trunc_pred,treshold) 

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

    def AP(self,cls,threshold):
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
            """
            if pred == 'TP':
                TP+=1
            recall.append(TP/TP_tot)
            if recall[-1]==recall[-2]:
                np = max(precision[-1],TP/(q+1))
                precision[-1] = np
                precision.append(np)
            else:
                precision.append(TP/(q+1))
            """
        print(recall,precision)
        plt.plot(recall,precision)
        plt.scatter(recall,precision)
        plt.xlim([0,1.2])
        plt.ylim([0,1.2])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Class: {cls}')       
        
        plt.show()
        return
            



if __name__ == '__main__':
    bina = MultiLabelBinarizer()
    x = [[5,2],[1,3,5],[4,0,1]]
    y = [[5,2],[3,1,3],[0,2]]

    X = bina.fit_transform(x)
    Y = bina.transform(y)
    #print(bina.classes_)
    #print(average_precision_score(X,Y))

    #true_boxes = np.array([[[100,50,300,150,1],[400,100,500,400,2],[150,50,300,100,3],[50,50,50,50,2]],[[100,50,300,150,1],[400,100,500,400,2],[150,50,300,100,3],[50,50,50,50,2]]])
    #pred_boxes = np.array([[[20,70,350,100,1,0.5],[400,100,450,400,2,0.7],[150,50,300,100,4,0.1]],[[20,70,350,100,1,0.5],[400,100,450,400,2,0.7],[150,50,300,100,4,0.1]]])
    true_boxes = np.array([[[0.3418472782258064,	0.41665123456790115,0.5569010416666667,	0.6460419155714855,	0],
                    [0.6196250560035843,	0.30514187574671436,	0.6913096438172042,	0.4548830147351651,	1],
                    [0.3741053427419354,	0.7033895858223815,	0.4708795362903226,	0.7989690362405415,	1],
                    [0.0031376008064516237,	0.5217886300278773,	0.28449960797491036,	0.9487101752289925,	2]],
                    [[0.368728998655914,	0.3847914177618478,	0.5712379592293907,	0.6587858422939067,	0],
                    [0.390234375,	0.7193194942254081,	0.46550319220430103,	0.8053409996017523,	1],
                    [0.6124565972222221,	0.3178858024691357,	0.6895175291218637,	0.45169703305455994,	1],
                    [0.0031376008064516237,	0.49630077658303456,	0.2701626904121864,	0.935966248506571,	2]]])

    pred_boxes = np.array([[[0.3418472782258064,	0.43576712465153317,	0.5676537298387097,	0.6460419155714855,	0,0.5],
                    [0.6052881384408602,	0.32744374751095173,	0.6805569556451612,	0.4293951612903225,	2,0.5],
                    [0.3543920810931899,	0.6683437873357226,	0.46371107750896046,	0.7671092194344883,	1,0.5],
                    [0.0031376008064516237,	0.4134652528872959,	0.2504494287634408,	0.9264083034647551,	2,0.5]],
                [[0.3723132280465949,	0.4230231979291119,	0.5712379592293907,	0.6046241537236161,	0,0.5],
                [0.3651447692652329,	0.6842736957387493,	0.49776125672043003,	0.8308288530465949,	1,0.5],
                [0.6232092853942651,	0.34655963759458375,	0.6895175291218637,	0.45169703305455994,	1,0.5],
                [0.008513944892473124,	0.5281605933890879,	0.13754620295698924,	0.7479933293508563,	2,0.5]]])
    maper = Mapper(true_boxes,pred_boxes,n_classes=3)
    maper.Check_Dataset()
    print(maper.counts)
    print(maper.Get_Metrics(1))
    maper.AP(1)
