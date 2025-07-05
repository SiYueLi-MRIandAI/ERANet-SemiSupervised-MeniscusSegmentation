import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import exposure, measure

class BaseOAI(Dataset):
    CLASSES = ['lateral_meniscus', 'lateral_tibial_cartilage', 'medial_meniscus', 'medial_tibial_cartilage']

    def __init__(self, base_dir, classes, csv_name, test_flag=True, height=256, width=256):
        self.root_dir = base_dir
        self.height = height
        self.width = width
        self.test_flag = test_flag
        self.class_values = [(self.CLASSES.index(cls.lower()) + 1) for cls in classes]
        self.image_list = pd.read_csv(csv_name, header=None)

    def __len__(self):
        return len(self.image_list)

    def histogram_normalize(self, image):
        p2, p98 = np.percentile(image, (10, 99))
        img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
        return img_rescale / (img_rescale.max() + 1e-6)

    def one_hot_encode(self, mask):
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=0).astype('float')
        if mask.shape[0] != 1:
            background = 1 - mask.sum(axis=0, keepdims=True)
            mask = np.concatenate((background, mask), axis=0)
        return mask

    def crop_center(self, image, label):
        x = (image.shape[0] - self.height) // 2 if self.test_flag else random.randint(0, image.shape[0] - self.height)
        y = (image.shape[1] - self.width) // 2 if self.test_flag else random.randint(0, image.shape[1] - self.width)
        return image[x:x+self.height, y:y+self.width], label[x:x+self.height, y:y+self.width], x, y

class OAIMeniscus(BaseOAI):
    def __init__(self, base_dir=None, classes=None, csv_name=None, test_flag=True):
        super().__init__(base_dir, classes, csv_name, test_flag, height=144, width=144)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_list.iloc[idx, 0])
        label_path = os.path.join(self.root_dir, self.image_list.iloc[idx, 1])
        name = os.path.basename(img_path)[4:]

        image = np.load(img_path).squeeze()
        label = np.load(label_path).squeeze()

        image = self.histogram_normalize(image)
        assert image.shape == label.shape

        image_cropped, label_cropped, x, y = self.crop_center(image, label)
        mri = np.load(img_path)[:, x:x+self.height, y:y+self.width]

        mask = self.one_hot_encode(label_cropped)
        img_tensor = torch.from_numpy(image_cropped.reshape(1, self.height, self.width).astype(np.float32))
        mri_tensor = torch.from_numpy(mri.astype(np.float32))
        mask_tensor = torch.from_numpy(mask).long()

        if self.test_flag:
            return {'image': img_tensor, 'label': mask_tensor, 'name': name, 'mri': mri_tensor}
        else:
            return {'image': img_tensor, 'label': mask_tensor}

class OAIMeniscusERA(BaseOAI):
    def __init__(self, base_dir=None, classes=None, csv_name=None, test_flag=True, random_factor=0.3):
        super().__init__(base_dir, classes, csv_name, test_flag, height=256, width=256)
        self.random_factor = random_factor

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_list.iloc[idx, 0])
        label_path = os.path.join(self.root_dir, self.image_list.iloc[idx, 1])
        name = os.path.basename(img_path)[4:-8]

        image = np.load(img_path)
        label = np.load(label_path)

        img = image.squeeze()
        label = label.squeeze()

        img = self.histogram_normalize(img)
        mask = self.one_hot_encode(label)
        mask_ori = mask.copy()

        meniscus_mask = mask[1] + mask[3]
        meniscus_mask = meniscus_mask[np.newaxis]
        connected_mask = measure.label(meniscus_mask, connectivity=2)
        components = measure.regionprops(connected_mask)

        cropped_mask = meniscus_mask.copy()
        zero_mask = np.zeros_like(meniscus_mask)
        restored_img = image.copy()

        if len(components) == 1:
            comp = components[0]
            _, y_min, x_min, _, y_max, x_max = comp.bbox
            length = x_max - x_min
            start_x = int(length * np.random.uniform(0, self.random_factor))
            end_x = int(length * np.random.uniform(0, self.random_factor))
            cropped_mask[:, :, x_min:x_min + start_x] = 0
            cropped_mask[:, :, x_max - end_x:x_max] = 0
        elif len(components) == 2:
            sorted_comps = sorted(components, key=lambda x: x.centroid[1])
            for comp in sorted_comps:
                _, y_min, x_min, _, y_max, x_max = comp.bbox
                length = x_max - x_min
                delta = int(length * np.random.uniform(0, self.random_factor))
                cropped_mask[:, :, x_min:x_min + delta] = 0
                cropped_mask[:, :, x_max - delta:x_max] = 0

        zero_mask = (meniscus_mask - cropped_mask)
        zero_mask[zero_mask != 0] = 1
        components_roi = measure.regionprops(measure.label(zero_mask, connectivity=2))

        for compo in components_roi:
            for _, y, x in compo.coords:
                restored_img[:, y, x] = np.random.randint(100, 201)

        if np.max(mask[1]) == 1:
            mask[1] = cropped_mask.squeeze()
        else:
            mask[3] = cropped_mask.squeeze()

        restored = restored_img.squeeze()
        restored_norm = self.histogram_normalize(restored)

        img_crop1, _, x, y = self.crop_center(restored_norm, label)
        img_crop2 = img[x:x+self.height, y:y+self.width]
        mask1 = mask[:, x:x+self.height, y:y+self.width]
        mask2 = mask_ori[:, x:x+self.height, y:y+self.width]

        return {
            'image_aug': torch.from_numpy(img_crop1[np.newaxis, ...].astype(np.float32)),
            'label_aug': torch.from_numpy(mask1).long(),
            'image': torch.from_numpy(img_crop2[np.newaxis, ...].astype(np.float32)),
            'label': torch.from_numpy(mask2).long()
        }
