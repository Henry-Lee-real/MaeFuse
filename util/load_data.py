import numpy as np
import os
import random
import cv2
import torch
from torch.utils.data import Dataset
from PIL import ImageEnhance
from PIL import Image



class MatchingImageDataset_mae(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(os.path.join(root_dir, "images"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "images", self.files[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

class MatchingImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.vi_files = os.listdir(os.path.join(root_dir, "vi"))
        self.ir_files = os.listdir(os.path.join(root_dir, "ir"))
        #self.label_files = os.listdir(os.path.join(root_dir, "label"))

        assert len(self.vi_files) == len(self.ir_files), \
        "Number of images in 'vi' and 'ir' directories must match."

    def __len__(self):
        return len(self.vi_files)

    def __getitem__(self, idx):
        vi_image_path = os.path.join(self.root_dir, "vi", self.vi_files[idx])
        ir_image_path = os.path.join(self.root_dir, "ir", self.ir_files[idx])
        #label_image_path = os.path.join(self.root_dir, "label", self.label_files[idx])

        vi_image = Image.open(vi_image_path)
        ir_image = Image.open(ir_image_path).convert('RGB')
        #label_image = Image.open(label_image_path).convert('RGB')
        
        #enhancer = ImageEnhance.Brightness(label_image)
        #label_image = enhancer.enhance(20.0)

        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)
            #label_image = self.transform(label_image)
 

        return vi_image, ir_image#, label_image

class MatchingImageDataset_GRAY(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.vi_files = os.listdir(os.path.join(root_dir, "vi"))
        self.ir_files = os.listdir(os.path.join(root_dir, "ir"))
        
        #self.ta_files = os.listdir(os.path.join(root_dir, "target"))

        assert len(self.vi_files) == len(self.ir_files), \
        "Number of images in 'vi' and 'ir' directories must match."

    def __len__(self):
        return len(self.vi_files)

    def __getitem__(self, idx):
        vi_image_path = os.path.join(self.root_dir, "vi", self.vi_files[idx])
        ir_image_path = os.path.join(self.root_dir, "ir", self.ir_files[idx])
        
        #ta_image_path = os.path.join(self.root_dir, "target", self.ta_files[idx])

        vi_image = Image.open(vi_image_path)
        vi_image_ycrcb = vi_image.convert('YCbCr')
        y_channel, cr_channel, cb_channel = vi_image_ycrcb.split()
        vi_image = y_channel.convert('RGB')
        ir_image = Image.open(ir_image_path).convert('RGB')
        
        # ta_image = Image.open(ta_image_path)
        # ta_image_ycrcb = ta_image.convert('YCbCr')
        # ta_y_channel, cr_channel, cb_channel = ta_image_ycrcb.split()
        # ta_image = ta_y_channel.convert('RGB')


        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)
            
            # ta_image = self.transform(ta_image)

        return vi_image, ir_image#, ta_image
    
class MatchingImageDataset_ONLY_SEG(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.vi_files = os.listdir(os.path.join(root_dir, "images"))
        self.ir_files = os.listdir(os.path.join(root_dir, "label"))

        assert len(self.vi_files) == len(self.ir_files), \
        "Number of images in 'vi' and 'ir' directories must match."

    def __len__(self):
        return len(self.vi_files)

    def __getitem__(self, idx):
        vi_image_path = os.path.join(self.root_dir, "images", self.vi_files[idx])
        ir_image_path = os.path.join(self.root_dir, "label", self.ir_files[idx])

        vi_image = Image.open(vi_image_path).convert('RGB')
        ir_image = Image.open(ir_image_path).convert('RGB')


        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)

        return vi_image, ir_image
    
class MatchingImageDataset_GRAY_SEG(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.vi_files = os.listdir(os.path.join(root_dir, "vi"))
        self.ir_files = os.listdir(os.path.join(root_dir, "ir"))
        self.label_files = os.listdir(os.path.join(root_dir, "label"))

        assert len(self.vi_files) == len(self.ir_files), \
        "Number of images in 'vi' and 'ir' directories must match."

    def __len__(self):
        return len(self.vi_files)

    def __getitem__(self, idx):
        vi_image_path = os.path.join(self.root_dir, "vi", self.vi_files[idx])
        ir_image_path = os.path.join(self.root_dir, "ir", self.ir_files[idx])
        label_image_path = os.path.join(self.root_dir, "label", self.label_files[idx])

        vi_image = Image.open(vi_image_path)
        vi_image_ycrcb = vi_image.convert('YCbCr')
        y_channel, cr_channel, cb_channel = vi_image_ycrcb.split()
        vi_image = y_channel.convert('RGB')
        ir_image = Image.open(ir_image_path).convert('RGB')
        label_image = Image.open(label_image_path).convert('RGB')
        # image_array = np.array(label_image)
        # image_array[image_array != 0] = 255
        # label_image = Image.fromarray(image_array)
        enhancer = ImageEnhance.Brightness(label_image)
        label_image = enhancer.enhance(30.0)

        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)
            label_image = self.transform(label_image)
 

        return vi_image, ir_image, label_image
    
class MatchingImageDataset_GRAY_Compare(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.vi_files = os.listdir(os.path.join(root_dir, "vi"))
        self.ir_files = os.listdir(os.path.join(root_dir, "ir"))
        self.ir_files = os.listdir(os.path.join(root_dir, "compare"))

        assert len(self.vi_files) == len(self.ir_files), \
        "Number of images in 'vi' and 'ir' directories must match."

    def __len__(self):
        return len(self.vi_files)

    def __getitem__(self, idx):
        vi_image_path = os.path.join(self.root_dir, "vi", self.vi_files[idx])
        ir_image_path = os.path.join(self.root_dir, "ir", self.ir_files[idx])
        compare_image_path = os.path.join(self.root_dir, "compare", self.ir_files[idx])

        vi_image = Image.open(vi_image_path)
        vi_image_ycrcb = vi_image.convert('YCbCr')
        y_channel, cr_channel, cb_channel = vi_image_ycrcb.split()
        vi_image = y_channel.convert('RGB')
        ir_image = Image.open(ir_image_path).convert('RGB')
        compare_image = Image.open(compare_image_path).convert('RGB')

        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)
            compare_image = self.transform(compare_image)
 

        return vi_image, ir_image, compare_image

class MatchingImageDataset_Multiple_exposure(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.vi_files = os.listdir(os.path.join(root_dir, "vi"))
        self.ir_files = os.listdir(os.path.join(root_dir, "ir"))
        #self.label_files = os.listdir(os.path.join(root_dir, "label"))

        assert len(self.vi_files) == len(self.ir_files), \
        "Number of images in 'vi' and 'ir' directories must match."

    def __len__(self):
        return len(self.vi_files)

    def __getitem__(self, idx):
        vi_image_path = os.path.join(self.root_dir, "vi", self.vi_files[idx])
        ir_image_path = os.path.join(self.root_dir, "ir", self.ir_files[idx])
        gt_image_path = os.path.join(self.root_dir, "gt", self.ir_files[idx])

        vi_image = Image.open(vi_image_path)
        ir_image = Image.open(ir_image_path)
        gt_image = Image.open(gt_image_path)

        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)
            gt_image = self.transform(gt_image)
 

        return vi_image, ir_image,gt_image

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def normalize_image(img,size=640):
    img = img / 255.
    assert img.shape == (size,size, 3)
    img = img - imagenet_mean
    img = img / imagenet_std
    return img

def prepare_image(img):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
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
        img_vi = cv2.imread(vi_path + name + vi_type, cv2.IMREAD_COLOR)
        img_ir = cv2.imread(ir_path + name + ir_type, cv2.IMREAD_COLOR)
        
        img_vi = cv2.cvtColor(img_vi, cv2.COLOR_BGR2RGB)   
        img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB)

        img_vi = cv2.resize(img_vi,(640,640),cv2.INTER_CUBIC) # type: ignore
        img_ir = cv2.resize(img_ir,(640,640),cv2.INTER_CUBIC) # type: ignore

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

