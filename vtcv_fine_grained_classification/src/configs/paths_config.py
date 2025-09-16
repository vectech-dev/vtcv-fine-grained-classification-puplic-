import os
from pathlib import Path
from pydantic import BaseModel
from vtcv_fine_grained_classification.src.configs.config import ExperimentationConfig


class PathsConfig(BaseModel):
    expconfig: ExperimentationConfig

    ''' def get_exp_dir(self, config=None, exp_name=None, path_format="experiments/{}"):
        if config is None and exp_name is None:
            raise ValueError("Either config or exp_name must be provided")
        if exp_name is None:
            exp_name = config.exp_name
        return path_format.format(exp_name) '''

    def get_experiment_datasheet_path(self, config=None, exp_name=None, path_format="experiments/{}/existing_data.csv"):
        if config is None and exp_name is None:
            raise ValueError("Either config or exp_name must be provided")
        if exp_name is None:
            exp_name = config.exp_name
        return path_format.format(exp_name)

    '''def get_save_dir(self, num_training=None):
        if not num_training:
            return self.expconfig.save_dir / self.expconfig.exp_name
        else:
            return Path(f"{self.expconfig.save_dir}/{self.expconfig.exp_name}/")

    # Define your get_img_abspath function here
    def get_img_abspath(self, image_id):
        img_path = self.expconfig.image_path
        return os.path.join(img_path, image_id)

    def get_staging_df_paths(self, staging_df_dir="data/staging_datasheets/"):
        staging_df_paths = []
        for file in os.listdir(staging_df_dir):
            if file.endswith(".csv"):
                staging_df_paths.append(os.path.join(staging_df_dir, file))
        return staging_df_paths '''
