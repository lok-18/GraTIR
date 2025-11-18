import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os

import torchvision.transforms as T


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

def sgg_img_process(img_path):
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    # images = []
    # images.append(img)
    img = transform(img)
    return img


class fusion_dataset(Dataset):
    def __init__(self, ir_path=None, vis_path=None):
        super(fusion_dataset, self).__init__()
        self.filepath_vis, self.filenames_vis = prepare_data_path(vis_path)
        self.filepath_ir, self.filenames_ir = prepare_data_path(ir_path)
        self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        # 定义图片大小调整的transform
        # self.resize = T.Resize((480, 640))


    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]

        # 用于场景图的输入
        image_vis_sgg = sgg_img_process(vis_path)
        image_ir_sgg = sgg_img_process(ir_path)

        # 读取并调整图片大小
        image_vis = Image.open(vis_path)
        # image_vis = self.resize(image_vis)
        image_vis = np.array(image_vis)
        
        image_inf = cv2.imread(ir_path, 0)
        image_inf = Image.fromarray(image_inf)
        # image_inf = self.resize(image_inf)
        image_inf = np.array(image_inf)
        
        image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
        )
        image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        image_ir = np.expand_dims(image_ir, axis=0)
        name = self.filenames_vis[index]
        # print(name)
        return (
            image_vis_sgg,
            image_ir_sgg,
            torch.tensor(image_vis),
            torch.tensor(image_ir),
            name,
        )

    def __len__(self):
        return self.length
        # return 8






