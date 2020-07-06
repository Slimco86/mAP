from dataset2 import *
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL.ImageDraw import Draw


if __name__ == '__main__':
    data_path = 'Z:\\Documents\\RoVis\\General_DataSet\\data_flow\\multi-class\\YOLO_DATA\\train2'

    training_params = {"batch_size": 2,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": collater,
                       "num_workers": 2}

    training_set = YoloDataset(root_dir=data_path,class_file="Z:\\Documents\\Repositories\\efficientdet-torch\\rovis.names",
                                transform=transforms.Compose([Resizer(common_size = 512),RandomGaussian(), Color_Jitter(),Normalizer()]))
    
    training_generator = DataLoader(training_set, **training_params)
    start = time.time()

    data = next(iter(training_generator))
    end = time.time()
    print(data['img'].size())
    img = transforms.ToPILImage()(data['img'][1])
    draw = Draw(img)
    for box in data['annot'][1].numpy():
        draw.rectangle(box[:4], fill=None, outline=None)
    img.show()
    print(end-start)
