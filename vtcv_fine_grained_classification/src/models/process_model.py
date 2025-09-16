import torch
import cv2
import os 
import numpy as np
import torch.nn.functional as F
torch.cuda.empty_cache() 
import time

import torch
import torch.nn.functional as F

from vtcv_fine_grained_classification.src.losses.losses import Diversity_loss
from vtcv_fine_grained_classification.src.visualization.attention_map import generate_gradcam



def process_topk(logits, config):
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)    
    prop, topk_cls = torch.topk(probabilities, k=config.k, dim=1)    
    return topk_cls,prop
    
def process_through_model(images, nets, config):
    if config.track_time:
        duration = {}
        start_time = time.time()
    
    # Process through base model
    logits, end_points = nets.base_model(images)
    end_points['logits'] = logits
    if config.track_time:
        duration["Pass_through_base_model"] = time.time() - start_time
        start_time = time.time()

    # Generate top-k predictions
    topk_cls, prop = process_topk(logits,  config)
    end_points["topk"] = topk_cls
    if config.track_time:
        duration["generate_topk"] = time.time() - start_time
        start_time = time.time()
    
    # Generate attention maps
    att_maps = []
    for i in range(config.k):
        if not config.single_network_model:
            att_map_emb = generate_gradcam(images, topk_cls[:,i], nets.emb_model)
            att_map_base = generate_gradcam(images, topk_cls[:,i], nets.base_model) 
            att_maps.append(nets.combine_attn_maps(torch.cat([att_map_base, att_map_emb],dim=1)))
        else:
            att_map_base = generate_gradcam(images, topk_cls[:,i], nets.base_model, save_dir="gradcam_outputs_top" + str(i + 1)) 
            att_maps.append(nets.combine_attn_maps(att_map_base))

    end_points['attention_map'] = F.interpolate(torch.cat(att_maps, dim =1), size=(config.imsize, config.imsize), mode='bicubic', align_corners=True, antialias=True) 
    
    if config.track_time:
        duration["attn_map_generation"] = time.time() - start_time
        start_time = time.time()

    diversity_label = torch.zeros(att_maps[0].size(0), dtype=torch.float32, device=att_maps[0].device)
    diversity_loss = Diversity_loss()
    loss_diversity = diversity_loss(att_maps[0], att_maps[1], diversity_label)

    if config.track_time:
        duration["loss_diversity_calculation"] = time.time() - start_time
        end_points["duration"] = duration

    end_points['loss_diversity'] = loss_diversity
    
    end_points["prop"] = prop


    return end_points


def process_through_model_cropping_dropping(images,  nets): 
    for i, image in enumerate(images):
        if (image == 0).all():
            # print(f"Image {i} is all zeros")
            pass
    logits, end_points = nets.base_model(images)
    end_points['logits'] = logits
    return end_points





''' # **Function to overlay Grad-CAM heatmap onto the input image**
def overlay_heatmap(image, cam):
    """
    Overlay Grad-CAM heatmap onto the original image.
    
    :param image: Original input image (C, H, W) in tensor format
    :param cam: Grad-CAM heatmap tensor (H, W)
    """
    cam = cam.cpu().detach().numpy()
    cam = cv2.resize(cam, (image.shape[1], image.shape[2]))  # Resize to match input image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # Apply color mapping
    heatmap = heatmap.astype(np.float32) / 255
    overlay = (0.5 * image.cpu().detach().numpy().transpose(1, 2, 0) + 0.5 * heatmap).clip(0, 1)  # Merge
    return overlay '''




