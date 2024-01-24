import numpy as np
import os
import random
import cv2
import torch

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def normalize_image(img):
    img = img / 255.
    assert img.shape == (224, 224, 3)
    img = img - imagenet_mean
    img = img / imagenet_std
    return img

def prepare_image(img):
    x = torch.tensor(img)
    #x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    return x.float()


def get_address(path):
    Path = path+"/vi"
    image_names = []

    for name in os.listdir(Path):
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tiff')):
            image_name = os.path.splitext(name)[0] 
            image_names.append(image_name)

    return image_names

def load_dataset(image_path, BATCH_SIZE):
    
    random.shuffle(image_path)
    batches = int(len(image_path) // BATCH_SIZE)
    return image_path, batches


def get_image_type(name,path):
    image_folder = path
    for type in ('.jpg', '.jpeg', '.png', '.gif', '.tiff'):
        image_filename = name + type 
        image_path = os.path.join(image_folder, image_filename)
        if os.path.exists(image_path):
            return type

def get_train_images(path ,names):

    vi_path = path + '/vi/'
    ir_path = path + '/ir/'
    
    vi_type = get_image_type(names[0],vi_path)
    ir_type = get_image_type(names[0],ir_path)
    
    VI = []
    IR = []

    for name in names:
        img_vi = cv2.imread(vi_path + name + vi_type)
        img_ir = cv2.imread(ir_path + name + ir_type)

        img_vi = cv2.resize(img_vi,(224,224),cv2.INTER_CUBIC)
        img_ir = cv2.resize(img_ir,(224,224),cv2.INTER_CUBIC)

        img_vi = normalize_image(img_vi)
        img_ir = normalize_image(img_ir)

        VI.append(img_vi)
        IR.append(img_ir)

    VI = np.stack(VI, axis=0)
    IR = np.stack(IR, axis=0)

    return prepare_image(VI) ,prepare_image(IR)



def load_labels(path ,names):
    labels_path = path + '/labels/'
    labels = []
    for i, name in enumerate(names):
        with open(labels_path + name + '.txt', 'r') as file:
            for line in file:
                out = [ float(data) for data in line.strip().split(" ")]
                out.insert(0, float(i))
                out = np.array(out)
                labels.append(out)
    
    labels = np.array(labels)
    labels = torch.tensor(labels)
    
    return labels

# name = get_address("MSRS-main/detection")
# image_path, batches = load_dataset(name,20)
# v,i = get_train_images("MSRS-main/detection",image_path)
# print(v.shape)
# print(type(v))
# load_labels("MSRS-main/detection",image_path)