from pydantic import BaseModel, Field
from typing import Any, ClassVar, Dict, List, Optional, Type
from vtcv_fine_grained_classification.src.losses.losses import FocalLoss, MosLoss




class ExperimentationConfig(BaseModel):

    """The experimentation configuration."""

    exp_name: str = Field(..., description="The name of the experiment")

    save_dir: str = Field(..., description="The directory for the saved models while training")

    mode: str = Field(
        default="train",
        description="The network mode i.e. `train` or `test` or `finetune`",
    )

    datasheet_path: Optional[str] = Field(..., description="The path to the datasheet")
    billy_datasheet_path: Optional[str] = Field(deafult=None, description="The path to the billy dataset")
    
    use_s3_bucket: Optional[bool] = Field(
        default=False,
        description="Whether or not to download data that doesn't exist locally from s3",
    )
 

    
    
    kwargs_augmentation: Dict[str, Any] = Field(..., description="The keyword arguments to the augmentation")
    whitebalance_cache_prefix: Optional[str] = Field(
        default="_wb",
        description="The suffix added to the databaseKey, producing the path to the cached whitebalanced images",
    )

    mask_kwargs: Dict[str, Any] = Field(default = {'prob_use_mask':0}, description="The keyword arguments to the mask")
    mask_cache_prefix: Optional[str] = Field(
        default="_masks",
        description="The suffix added to the databaseKey, producing the path to the cached masks",
    )
    

    track_time: bool = Field(default = False, description = "Enables time tracking for various stages of training")
    epochs: Optional[int] = Field(default=20, description="The number of epochs when training")
    test_epoch: Optional[int] = Field(default=20, description="Which epoch to test")
    batch_size: int = Field(default=64, description="The batch size when training")
    num_workers: int = Field(default=14, description="The number of workers to use in dataloaders")
    lr : float = Field(default=0.1, description="The learning rate for adam optimizer")
    weight_decay: float = Field(default=1e-3, description="The weight decay for the optimizer")
    


    backgournd_RGB_cropped: List[int] = Field(default=[255, 0, 255], description="RGB values for the cropped image background")
    attn_model: Optional[str] = Field(
        default='bap',
        description="Which attention model to use: bap, multihead",
    )
    k : int = Field(default=2, description="The number of topk")
    imsize: int = Field(default=299, description="The image size")
    max_pixel: int = Field(default=255, description="Maximum pixel value")
    min_pixel: int = Field(default=0, description="Minimum pixel value")
    num_classes: int = Field(default=32, description="The number of classes")
    one_hot_labels: bool = Field(default=False, description="Whether to use one hot labels")

    single_network_model: bool = Field(default=False, description = "Whether to use separate base and taxonomy networks, or if the base model be pretrained with taxonomy network parameters values")
    freeze_ratio_single_model: float = Field(default=0.0, description="Ratio of the single network parameters to freeze during training")
    initial_model_kwargs: Dict[str, Any] = Field(default = {}, description = "Keyword arguments for loading initial parameter values for the taxonomy and base models")
    freeze_weights_kwargs: Dict[str, Any] = Field(default = {},
                                                  description = "Keyword arguments for freezing taxonomy model parameters")
    debug : bool = Field(default=False, description="Whether to run in debug mode")       
    early_stopping_kwargs: Dict[str, Any] = Field(default={}, description="The keyword arguments to the early stopping")
    save_every : int = Field(default=50, description="The number of epochs to save the model")


    # [logits_image, logits_crop, logits_drop, logits_topk]
    weight_logits_base: float = Field(default=0.33, description="The weight for the image loss")
    weight_logits_crop: float = Field(default=0.33, description="Weight for the crop loss")
    weight_logits_drop: float = Field(default=0.33, description="Weight for the drop loss")
    threshold_ratio_crop: float = Field(default=0.1, description="Hyperparameter to generate adaptive threshold for crop")
    threshold_ratio_drop: float = Field(default=0.9, description="Hyperparameter to generate adaptive threshold for drop")
    num_batches_att_visualization_valid: int = Field(default=0, description="How many batches of attention visualization images to save for validation dataset (for example: first b batches will be saved per epoch)")
    num_batches_att_visualization_test: int = Field(default=0, description="How many batches of attention visualization images to save for test dataset (for example: first b batches will be saved per epoch)")
    losses: ClassVar[Dict[str, Type]] = {
        "MosLoss": MosLoss,
        "FocalLoss": FocalLoss
    }
    loss: str = Field(..., description="The loss to use")
    loss_kwargs: Optional[Dict] = Field(default={}, description="The keyword arguments for the loss")
    weighted_loss_parameters: Optional[Dict] = Field(default = {"combined_loss_weight": 0.7,
                                                                "contrastive_loss_weight": 0.3},
                                                                description = "Keyword arguments for weighted loss")
   

    device: str = Field(default="cuda", description="The device to use")
    gpu_ids: List[int] = Field(default=[0], description="The gpu ids to use")                           






    #lr_decay: float = Field(default=0.95, description="The learning rate decay")
    
    ''' mask_path: Optional[str] = Field(
        default="/opt/ImageBase/mosID-production_masks/",
        description="The path to the mask",
    ) '''

    ''' initial_model_weights_path: Optional[str] = Field(
        default=None,
        description="The path to the initial model weights when finetuning",
    ) '''
    
    ''' emb_model: Optional[str] = Field(
        default='lstm',
        description="Which embedding model to use: lstm, simple, static, node2vec",
    ) '''

    #emb_dim : int = Field(default=50, description="The embedding dimension")

    #rnn_size : int = Field(default=768, description="The rnn size")
    
    ''' save_dir: Optional[str] = Field(
        default=None,
        description="The directory for the saved models while training",
    ) '''

    #simple_model: bool = Field(default=False, description="Excludes crop and drop if true")

    #beta1 : float = Field(default=0.9, description="The beta1 for adam optimizer")
    #beta2 : float = Field(default=0.999, description="The beta2 for adam optimizer")
    #checkpoint_dir : str = Field(default="checkpoints", description="The directory for the checkpoints")
    #num_layers : int = Field(default=1, description="The number of layers")
    #dropout : float = Field(default=0.5, description="The dropout")
    #dataset: str = Field(default="MosMaskDataset", description="The dataset to use")


    
    
