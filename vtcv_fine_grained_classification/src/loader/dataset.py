import os
import torch
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from vtcv_fine_grained_classification.src.loader.mask import Mask



class SquaredImage:
    def __init__(self,val:int=-1):
        self.val=val

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.make_square(img,self.val)

    @staticmethod
    #EDITTED
    def make_square(img,val):
        img = np.array(img)
        rolled=False
        if img.shape[0] > img.shape[1]:
            img = np.rollaxis(img, 1, 0)
            rolled=True
        toppadlen = (img.shape[1] - img.shape[0]) // 2
        bottompadlen = img.shape[1] - img.shape[0] - toppadlen
        if val == -1:
            toppad = img[:5, :, :].mean(0, keepdims=True).astype(img.dtype)
            toppad = np.repeat(toppad, toppadlen, 0)
            bottompad = img[-5:, :, :].mean(0, keepdims=True).astype(img.dtype)
            bottompad = np.repeat(bottompad, bottompadlen, 0)
        else:
            toppad=val*np.ones((toppadlen,img.shape[1],3),dtype=np.uint8)
            bottompad=val*np.ones((bottompadlen,img.shape[1],3),dtype=np.uint8)
            
        img = np.concatenate((toppad, img, bottompad), axis=0)
        if rolled:
            img = np.rollaxis(img, 1, 0)
        return Image.fromarray(np.uint8(img))



class CV2Resize:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        # Convert PIL Image to numpy array (H x W x C)
        img_np = np.array(img)
        # Resize using cv2
        img_resized = cv2.resize(img_np, self.size, interpolation=cv2.INTER_AREA)       
        # Convert numpy array back to PIL Image
        return Image.fromarray(img_resized)

class CustomCombinedTransform:
    def __init__(
        self,
        config,
        imsize,
        masksize,
        mode,
        p,
        setting,
        save_preprocessed,
    ):
        self.config = config
        self.imsize = imsize
        self.mode = mode
        self.p = p
        self.setting = setting
      
        self.masksize = masksize
        self.normalize_constants={'mean':[0.5, 0.5, 0.5], 
                                  'std':[0.5, 0.5, 0.5]}
        self.alb_transform_train = self.alb_transform_train(self.p, self.setting,self.imsize)
        self.alb_degrade = self.alb_degrade(self.p, self.setting,self.imsize)
     
        self.tv_transform_mask = self.tv_transform_mask(self.masksize)
        self.tv_transform = self.tv_transform(self.imsize,self.normalize_constants)
        self.square = self.square()
        # self.inv_normalize=self.inv_normalize(self.normalize_constants)

    @staticmethod
    def alb_transform_train(p=1, setting=0,imsize=448):
        if setting is None:
            setting = 0
         
        if setting in [1055,1050,1045,1040,1035,1030,1025,1020,1015,1010,1005]:
            setting = 0
        """
        settings 10xx are made for image decimation experimentation to approximate the effect 
        of lower resolutions in the IDX dataset. xx in 10xx represents the approximate percentage
        of resultion degradation from the full resolution. This is based on an assumption of 
        875x875 pizel size full resolution images at 36 lp/mm. the morphological operations added
        are determined in this googlesheet:   
        https://docs.google.com/spreadsheets/d/14B4-wUmrBGJF1wsOrfGQXDbNUPnOBduq5BvxkghNdRI/edit?gid=0#gid=0
        """
        if setting == 0:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    HorizontalFlip(),
                    VerticalFlip(),
                    Transpose(),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=0.01),
                    ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
                    ),
                ],
                p=1,
            )
        if setting == 1:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    HorizontalFlip(),
                    VerticalFlip(),
                    Transpose(),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                    ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
                    ),
                ],
                p=1,
            )
        elif setting == 2:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    HorizontalFlip(),
                    VerticalFlip(),
                    Transpose(),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                    ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
                    ),
                ],
                p=1,
            )
        elif setting == 3:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    HorizontalFlip(),
                    VerticalFlip(),
                    Transpose(),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                    ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
                    ),
                ],
                p=1,
            )
        elif setting == 4:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    HorizontalFlip(),
                    VerticalFlip(),
                    Transpose(),
                    ColorJitter(
                        brightness=0.15, contrast=0, saturation=0.1, hue=0.025, p=0.7
                    ),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                    ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
                    ),
                ],
                p=1,
            )
        elif setting == 5:
            albumentations_transform = Compose(
                [
                    RandomRotate90(),
                    HorizontalFlip(),
                    VerticalFlip(),
                    Transpose(),
                    ColorJitter(
                        brightness=0.15, contrast=0, saturation=0.1, hue=0.1, p=0.7
                    ),
                    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=p),
                    ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
                    ),
                ],
                p=1,
            )
        elif setting == 30:
            albumentations_transform = Compose(
                [
                    ColorJitter(
                        brightness=0.15, contrast=0, saturation=0.1, hue=0.1, p=0.7
                    ),
                ],
                p=1,
            )
            
        # Apply the transformation on the image
        def apply_transform(image):
            transformed = albumentations_transform(image=image)
            return transformed["image"]

        return apply_transform

    @staticmethod
    def alb_degrade(p=1, setting=0,imsize=448,degrade_key={1055: 481,
                1050: 437,
                1045: 394,
                1040: 350,
                1035: 306,
                1030: 262,
                1025: 218,
                1020: 175,
                1015: 131,
                1010: 87,
                1005: 43}):
        """
        settings 10xx are made for image decimation experimentation to approximate the effect 
        of lower resolutions in the IDX dataset. xx in 10xx represents the approximate percentage
        of resultion degradation from the full resolution. This is based on an assumption of 
        875x875 pizel size full resolution images at 36 lp/mm. the morphological operations added
        are determined in this googlesheet:   
        https://docs.google.com/spreadsheets/d/14B4-wUmrBGJF1wsOrfGQXDbNUPnOBduq5BvxkghNdRI/edit?gid=0#gid=0
        """
        if setting in degrade_key:
            degrade_size=degrade_key[setting]
        
            albumentations_transform = Compose(
                [
                    Resize(height=degrade_size,width=degrade_size,interpolation=cv2.INTER_AREA,p=1) ,
                    GaussianBlur(blur_limit=(5,5),p=1),
                    GaussNoise(var_limit=(25,25), always_apply=True, p=1),
                    Resize(height=imsize,width=imsize,interpolation=cv2.INTER_AREA,p=1) ,
                ],
                p=1,
            )
        else:
            albumentations_transform = Compose([],p=1,)

        # Apply the transformation on the image
        def apply_transform(image):
            transformed = albumentations_transform(image=image)
            return transformed["image"]

        return apply_transform



    def tv_transform_mask(self, masksize=448):
        tv_transform_mask = T.Compose(
            [
                SquaredImage(215),
                CV2Resize((masksize, masksize)),
                T.ToTensor(),
            ]
        )

        return tv_transform_mask

    def tv_transform(self, imsize=299,normalize_constants={'mean':[0.5, 0.5, 0.5], 
                                  'std':[0.5, 0.5, 0.5]}):
        tv_transform = T.Compose(
            [
                T.ToPILImage(),
                CV2Resize((imsize, imsize)),
                T.ToTensor(),
                T.Normalize(mean=normalize_constants['mean'], std=normalize_constants['std']),
            ]
        )

        return tv_transform

class MosMaskDataset(Dataset):
    def __init__(self, config, data_df, num_classes, mode, transform=None, one_hot_label=False):
        """
        Params:
            data_df: data DataFrame of image name and labels
            transform: optional data transformer
            one_hot_label: whether to return one-hot encoded labels
            mode: train or test/validation
            do_mask: whether to apply mask to image
        """
        super().__init__()
        self.config = config
        self.exp_config = config.expconfig
        self.mode = mode
        self.num_classes = num_classes
        self.transform = transform
        self.data_df = data_df  
        self.known_unique_species=None
        self.unknown_unique_species=None
        self.database_keys = self.exp_config.kwargs_s3_bucket.database_keys
        self.images_df = pd.read_csv(data_df, dtype={"Specimen_Id":str,"Id":str,"Well": str, "FeedingStatus": str})     
        self.prep_data_sheet()
        self.one_hot_label = one_hot_label      
        self.mask = Mask(config, **self.exp_config.mask_kwargs)
        self.masksize = self.exp_config.mask_kwargs["mask_size"]
        self.imsize = self.exp_config.imsize
        self.kwargs_augmentation = self.exp_config.kwargs_augmentation
       
        self.transform = CustomCombinedTransform(
            self.config,
            imsize=self.imsize,
            masksize=self.masksize,
            mode=self.mode,
            **self.kwargs_augmentation,
        )


    def __len__(self) -> int:
        return len(self.images_df)

    def __getitem__(self, idx: int) -> tuple:
        try:
            current_sample = self.images_df.iloc[idx]
            img_path = current_sample["Id"]
          
            label = current_sample["y"]
           
           
            if self.kwargs_augmentation['white_balance']:
                wb_img_path = current_sample["White_Balance_Path"]              
                if not os.path.exists(wb_img_path):
                    os.makedirs(os.path.dirname(wb_img_path), exist_ok=True)
                    img = Image.open(img_path).convert("RGB")  # Load image
                    # Save the transformed image to the specific path
                    img.save(wb_img_path)
                else:
                    img = Image.open(wb_img_path).convert("RGB")
            else:
                img = Image.open(img_path).convert("RGB")  # Load image

            img = self.transform.square(img)
            img = np.array(img)
       
        except Exception as e:
            return self.__getitem__((idx + 1) % self.__len__())

        if img is None:
            return self.__getitem__((idx + 1) % self.__len__())        
               
        
        # Convert label to one-hot encoding if specified
        if self.one_hot_label:
            one_hot_label = torch.zeros(self.num_classes)
            one_hot_label[label] = 1
            label = one_hot_label 

        
        transforms_ = transforms.Compose([              
                CV2Resize((self.exp_config.imsize, self.exp_config.imsize)),]) 
        
        img_transformed = transforms.Compose([
                transforms.ToTensor(),
              
            ])        
        img = transforms_(img) 
        mask = self.mask.load_mask(img, current_sample, self.transform)
        mask =cv2.resize(np.array(mask), (self.imsize, self.imsize))
        img = np.array(img)
        img = self.mask.apply_mask(img, mask, self.mode)        
        if self.mode == 'Train':        
            img = img_transformed(img)           
        else: 
            img = img_transformed(img)
      
        return img, label
    

    def prep_data_sheet(self, saved_path='tmp', classify_column = "y", max_num_per_cls_train = 250, max_num_per_cls_val = 50):
        self.images_df = self.images_df[(self.images_df[classify_column] != -1) & (self.images_df["Split"] == self.mode)].copy()
        self.images_df = self.images_df[self.images_df['Id'].apply(lambda x: os.path.exists(x))]
        self.images_df["Mask_Path"] = pd.NA
        self.images_df["White_Balance_Path"] = pd.NA
        
        # prep mask & wb paths
        for i, row in self.images_df.iterrows():
            database = row["Database"]
            self.images_df.loc[i, "Mask_Path"] = mask_path
            self.images_df.loc[i, "White_Balance_Path"] = wb_img_path

        if not os.path.exists(saved_path):
            os.makedirs(saved_path, exist_ok=True)
    
        if self.mode == 'Train':
            self.images_df = (
                self.images_df.groupby(classify_column, group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), max_num_per_cls_train), random_state=42))  # Take all if < 250
                .reset_index(drop=True)
            )

        else: 
            self.images_df = (
                self.images_df.groupby(classify_column, group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), max_num_per_cls_val), random_state=42))  # Take all if < 50
                .reset_index(drop=True)
            )
        print(f"Saving {self.mode} dataset to {saved_path}", len(self.images_df))
        self.images_df.to_csv("{}/current_run_{}_dataset.csv".format(saved_path, self.mode), index=False)

    def get_labels(self, classify_column = "y"):
        return self.images_df[classify_column]

