# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from .network import MFF_MoE
import folder_paths
from comfy.utils import ProgressBar

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
current_path = os.path.dirname(os.path.abspath(__file__))

def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def images_generator(file_list):# form VH
    sizes = {}
    for i in file_list:
        count = sizes.get(i.size, 0)
        sizes[i.size] = count +1
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    total_images = len(file_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(phi2narry, file_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image

def load_images(file_list):
    gen = images_generator(file_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded from directory '{file_list}'.")
    return images

class DeepFakeDefender_Loader:
    def __init__(self):
       pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_path": ("STRING", {"default": "DeepFakeDefender"}),
            }
        }

    RETURN_TYPES = ("MODEL","MODEL",)
    RETURN_NAMES = ("net","transform_val",)
    FUNCTION = "test"
    CATEGORY = "DeepFakeDefender_Gold"

    def test(self,ckpt_path):
        weigths_current_path = os.path.join(folder_paths.models_dir, ckpt_path)
        if not os.path.exists(weigths_current_path):
            os.makedirs(weigths_current_path)
        net = MFF_MoE(pretrained=False)
        net.load(path=weigths_current_path)
        net = nn.DataParallel(net).cuda()
        net.eval()
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 512), antialias=True),
        ])
        return (net,transform_val,)
    
class DeepFakeDefender_Sampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "net": ("MODEL",),
                "transform_val": ("MODEL",),
                "threshold": ("FLOAT", {
                    "default": 0.500000000,
                    "min": 0.000000001,
                    "max": 0.999999999,
                    "step": 0.001,
                    "round": 0.0000000001,
                    "display": "number",
                }),
            },
        }
    
    RETURN_TYPES = ("STRING","IMAGE","IMAGE",)
    RETURN_NAMES = ("string","above","below",)
    FUNCTION = "test"
    CATEGORY = "DeepFakeDefender_Gold"
    
    def test(self,image,net,transform_val,threshold):
        B, _, _, _ = image.shape
        if B==1:
            img_list=[tensor2pil(image)]
        else:
            img_list = list(torch.chunk(image, chunks=B))
            img_list = [tensor2pil(img) for img in img_list]
        out_str=''
        below_img=[]
        above_img=[]
        pred_check=np.array([threshold])
        for i in range(B):
            x = img_list[i]
            # x = cv2.imread(input_path)[..., ::-1]
            # x=np.array(x)
            #x=np.asarray(x)
            # x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)[..., ::-1]
            #x = Image.fromarray(np.uint8(x))
           
            y = transform_val(x).unsqueeze(0).cuda()
            pred = net(y)
            pred = pred.detach().cpu().numpy()
            print('Prediction of this image [%s] being Deepfake(这张照片是深度伪造的预测值为): %10.9f' % (i+1,pred))
            pred = np.around(pred, decimals=9)
            string = f"\nPrediction of this image ({i+1}) being Deepfake: {pred}. \n 这张照片({i+1})是深度伪造的预测值为：{pred}.\n"
            out_str += string
            
            if np.less_equal(pred,pred_check)==[True] :
                below_img.append(x)
            if np.greater(pred,pred_check)==[True] :
                above_img.append(x)
        if below_img:
            below_image=load_images(below_img)
        if above_img:
            above_image = load_images(above_img)
        return (out_str,above_image,below_image)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DeepFakeDefender_Loader": DeepFakeDefender_Loader,
    "DeepFakeDefender_Sampler":DeepFakeDefender_Sampler
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepFakeDefender_Loader": "DeepFakeDefender_Loader",
    "DeepFakeDefender_Sampler":"DeepFakeDefender_Sampler"
}
