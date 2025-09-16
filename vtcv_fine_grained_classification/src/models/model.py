import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time
from loguru import logger
from pretrainedmodels import xception
from vtcv_fine_grained_classification.train.train_utils import model_forward_pass, compute_loss



# Logger setup
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")







class SEBlock(nn.Module):
    def __init__(self, channels):
        super(SEBlock, self).__init__()
        # Example placeholder
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // 16)
        self.fc2 = nn.Linear(channels // 16, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class Xception_V2_1(nn.Module):
    def __init__(self, num_classes=32, stop_layer='block11'):
        super(Xception_V2_1, self).__init__()

        # 1) Load full pretrained Xception
        backbone = xception(pretrained=None)

        # 2) Overwrite last_linear to match (728 → num_classes)
        backbone.last_linear = nn.Linear(in_features=728, out_features=num_classes, bias=True)

        # 3) Store the entire model
        self.model = backbone

        # 4) SE Block
        self.se_block = SEBlock(728)

        # 5) Extract block11 explicitly 
        #    (so it is registered as a submodule in THIS class).
        self.block11 = self.model.block11

        # Optionally remove or replace the original block11 reference
        # so that you don't accidentally run it twice.
        # This is optional, but typically you don't want two copies.
        self.model.block11 = nn.Identity()
        
        self.stop_layer = stop_layer
        self.initialize_weights()
       
    
    def initialize_weights(self, pretrained=None):
        if pretrained is None:
            # Initialize only if not using pretrained weights
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        # Always initialize the SE block and last linear layer
        for m in self.se_block.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.xavier_uniform_(self.model.last_linear.weight)
        if self.model.last_linear.bias is not None:
            nn.init.constant_(self.model.last_linear.bias, 0)
        
        

    def forward(self, x, y=None):
        """
        Manually run everything up to block10, then run self.block11,
        then run your SE block, etc.
        """

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)

        # Now do blocks 1..10
        x = self.model.block1(x)
        x = self.model.block2(x)
        x = self.model.block3(x)
        x = self.model.block4(x)
        x = self.model.block5(x)
        x = self.model.block6(x)
        x = self.model.block7(x)
        x = self.model.block8(x)
        x = self.model.block9(x)
        x = self.model.block10(x)

        # Stop layer is 'block11' → call self.block11
        ftm = self.block11(x)       # Feature maps at block11

        # Apply your SE Block
        x = self.se_block(ftm)

        # Adaptive pooling + flatten
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.model.last_linear(x)

        # Collect intermediate features
        end_points = {'feature_maps': ftm}

        return x, end_points




class MainModel(torch.nn.Module):
    def __init__(self, nets, losses, config, exp_dir=None):
        super(MainModel, self).__init__()
        self.nets = torch.nn.ModuleDict(nets)  
        self.losses = losses
        self.config = config
        self.exp_dir = exp_dir
        self.index_epoch = 1
        self.index_batch = 1
        self.device = config.device
        
    def forward(self, images,  labels, mode, cur_epoch = 0, cur_batch = 0):
        end_points = model_forward_pass(self.config, images, labels,  self.nets, cur_epoch, cur_batch, mode=mode)
        if self.config.track_time:
            duration = end_points["duration"]
            start_time = time.time()
        criterion_loss= compute_loss( self.losses, labels, self.config, end_points=end_points)
        if self.config.track_time:
            duration["criterion_loss_calculation"] = time.time() - start_time
            end_points["duration"] = duration

        return criterion_loss, end_points
    

