import sys
import os
import pandas as pd
import torch.utils.data
from munch import Munch

from typing import Callable
from vtcv_fine_grained_classification.src.configs.config import ExperimentationConfig
from vtcv_fine_grained_classification.src.configs.paths_config import PathsConfig
from vtcv_fine_grained_classification.src.models.build_model import build_model
from vtcv_fine_grained_classification.src.loader.dataset import MosMaskDataset




class TrainConfig(BaseModel):

    """The training configuration."""

    expconfig: ExperimentationConfig
    paths_config: PathsConfig

    def get_model(self) -> torch.nn.Module:
        return self.expconfig.models[self.expconfig.model_name]

    def get_num_classes(self, path, classify_column = "y", column = "Species_Name"):
        return len(self.get_species_names(path = path, classify_column = classify_column, column = column))

    def get_species_names(self, path, classify_column = "y", column = "Species_Name"):
        if path is None:
            sys.exit("No path is provided")
        df = pd.read_csv(path, low_memory=False)
        classes = []
        df = df[df.Split == "Train"]
        ys = df[classify_column].unique()
        class_map = {}
        for y in range(max(ys) + 1):
            class_map[y] = df[column].loc[df[classify_column] == y].values[0]
            classes.append(class_map[y])
        return classes

    def call_exp_paths(self, x=None):
        paths = self.paths_config
        if x is not None:
            return paths.get_img_abspath(x)
        else:
            return paths.get_experiment_datasheet_path(exp_name=self.expconfig.exp_name)

    def get_dataset(
        self,
        mode: str,
        num_classes: int = None,
        data_override: str = None,
        staging: bool = False,
    ) -> MosMaskDataset:
        if data_override is not None and mode == "Test":
            data_fn = self.save_processed_datasets(data_override=data_override)
            num_classes = num_classes   
            dataset = MosMaskDataset(config=self, data_df=data_fn, num_classes=num_classes, mode=mode)
            print("Dataset loaded from override path *********************************************", len(dataset))
            return dataset, data_fn
        else:
            data_fn = self.paths_config.get_experiment_datasheet_path(exp_name=self.expconfig.exp_name)
            if not os.path.exists(data_fn):
                data_fn = self.save_processed_datasets()
            num_classes = self.get_num_classes(data_fn)

        dataset = MosMaskDataset(config=self, data_df=data_fn, num_classes=num_classes, mode=mode)
        if staging:
            return dataset, data_fn
        return dataset   


    def get_loss(self, num_classes, device) -> Callable:
        if self.expconfig.loss == "FocalLoss":
            [v.pop("weight") for v in self.expconfig.loss_kwargs["loss_dict"].values()]
            [v.pop("mag_scale") for v in self.expconfig.loss_kwargs["loss_dict"].values()]
            return self.expconfig.losses[self.expconfig.loss](num_classes, **(list(self.expconfig.loss_kwargs["loss_dict"].values())[0])).to(device)
        else:
            return  self.expconfig.losses[self.expconfig.loss](num_classes, self.expconfig.loss_kwargs["loss_dict"]).to(device)

    def build_model(self, config, device,  num_classes):
        nets = build_model(config, device,  num_classes)
       
        return nets
    

    
    