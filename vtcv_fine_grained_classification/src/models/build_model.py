from munch import Munch
from vtcv_fine_grained_classification.src.models.model import Xception_V2_1
import torch.nn as nn

def build_model(config, device, num_classes):
    # Initialize the models
    if not config.single_network_model:
        base_model =Xception_V2_1(num_classes).to(device) 
        emb_model  = Xception_V2_1(num_classes).to(device)
        combine_attn_maps = nn.Sequential(
                        nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
                        nn.LeakyReLU()
                    )
        nets = Munch(base_model=base_model, emb_model=emb_model, combine_attn_maps=combine_attn_maps)
    else:
        base_model =Xception_V2_1(num_classes).to(device)
        combine_attn_maps = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
                        nn.LeakyReLU()
                    )
        nets = Munch(base_model=base_model, combine_attn_maps=combine_attn_maps)
         
    return nets


def define_losses(criterion_class):
    losses = Munch()
    losses.criterion = criterion_class
    return losses
    

