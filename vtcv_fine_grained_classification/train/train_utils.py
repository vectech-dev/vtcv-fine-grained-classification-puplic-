import torch
import time

from vtcv_fine_grained_classification.src.models.region import *
from vtcv_fine_grained_classification.src.models.region import attention_crop, attention_drop
from vtcv_fine_grained_classification.src.visualization.visual_region import  crop_images
from vtcv_fine_grained_classification.src.utils.utils import save_images_and_att_maps
from vtcv_fine_grained_classification.src.visualization.visualize_att_map import create_and_save_subplot
from vtcv_fine_grained_classification.src.models.process_model import process_through_model,  process_through_model_cropping_dropping
import time



def compute_loss(losses, labels, config, end_points):
    loss_crop, loss_drop =  0, 0
    loss_base = losses.criterion(end_points['logits'], labels)
    if config.weight_logits_crop != 0:
        loss_crop = losses.criterion( end_points['logits_crop'], labels)
    else:
        loss_crop = loss_base
    if config.weight_logits_drop != 0 :
        loss_drop = losses.criterion(end_points['logits_drop'] , labels)
    else:
        loss_drop = loss_base
    end_points["combined_logits"] = config.weight_logits_crop * end_points['logits_crop'] \
            + config.weight_logits_drop * end_points['logits_drop']  + config.weight_logits_base * end_points['logits']
    combined_loss =  losses.criterion(end_points["combined_logits"], labels) 
  

    criterion_losses = [loss_base, loss_crop, loss_drop, combined_loss]
  
    return criterion_losses







def add_noise(tensor, cur_epoch, config, percent = 0.35):
    num_elements = tensor.numel()
    num_noise_cells = int(num_elements * percent/100)
    noisy_indices = torch.randperm(num_elements)[:num_noise_cells]
    noise = torch.zeros_like(tensor).flatten()
    noise = noise.to(tensor.device)
    num_noise_cells = noisy_indices.to(tensor.device).shape[0]
    noise_values = torch.randn(num_noise_cells, device=noise.device)
    noise[noisy_indices] = noise_values
    noise = noise.view(tensor.shape)
    noisy_tensor = tensor + noise * (cur_epoch/config.epochs)
    noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
    return noisy_tensor.to(tensor.device)



def model_forward_pass(config, images, labels, nets, cur_epoch=0, cur_batch=0, mode='Train'):
    """
    A combined forward pass that:
      - Adds noise if mode == 'Train'
      - Runs the base model
      - Optionally applies attention cropping / dropping
      - If mode == 'Valid', saves a subplot per image in label-specific folders
    """

    # TRAIN LOGIC
    if mode == 'Train':
        images = add_noise(images, cur_epoch, config)

    # Normal forward pass
    end_points = process_through_model(images, nets, config)

    attention_maps_resized = end_points['attention_map']  # shape (B, k, H, W), e.g. k=2



    ##
    # Attention Crop
    if config.weight_logits_crop != 0:
        if config.track_time:
            duration = end_points["duration"]
            start_time = time.time()

        bboxes = attention_crop(attention_maps_resized, config.threshold_ratio_crop)
        end_points['bboxes'] = bboxes  
        images_crop = crop_images(images, bboxes.clone().to(images.device), config)

        # Forward pass on cropped images
        end_points_crop = process_through_model_cropping_dropping(images_crop, nets)
        end_points['logits_crop'] = end_points_crop['logits']

        if config.track_time:
            duration["creating_and_processing_crop"] = time.time() - start_time
            end_points["duration"] = duration

    else:
        images_crop = images
        end_points['logits_crop'] = end_points['logits']
    



    # Attention Drop
    if config.weight_logits_drop != 0:
        if config.track_time:
            duration = end_points["duration"]
            start_time = time.time()

        images_drop = attention_drop(images, attention_maps_resized, config.threshold_ratio_drop)

        # Forward pass on dropped images
        end_points_drop = process_through_model_cropping_dropping(images_drop, nets)
        end_points['logits_drop'] = end_points_drop['logits']

        if config.track_time:
            duration["creating_and_processing_drop"] = time.time() - start_time
            end_points["duration"] = duration

    else:
        images_drop = images
        end_points['logits_drop'] = end_points['logits']


    # Save subplots for validation
    ''' if (mode == 'Valid' or mode=='Test') and config.attention_visualization_for_valid:
        end_points['images_crop'] = images_crop
        end_points['images_drop'] = images_drop

        val_data_for_visualization = save_images_and_att_maps(images, labels, end_points)
        if cur_epoch % config.save_every == 0:
            # Produce the subplots with original, cropped, dropped, attention maps
            create_and_save_subplot(val_data_for_visualization, config, "images_valid/" + config.exp_name) '''

    if (mode == 'Valid' and config.num_batches_att_visualization_valid > cur_batch) or (mode == 'Test' and config.num_batches_att_visualization_test > cur_batch):
        end_points['images_crop'] = images_crop
        end_points['images_drop'] = images_drop

        data_for_visualization = save_images_and_att_maps(images, labels, end_points)
        if mode == 'Valid':
            save_dir = "experiments/" + config.exp_name + "/visualizations_valid"
            batch_id_for_image_name = cur_batch % config.num_batches_att_visualization_valid
        else:
            save_dir = "experiments/" + config.exp_name + "/test_predictions/visualizations"
            batch_id_for_image_name = cur_batch % config.num_batches_att_visualization_test
        # Produce the subplots with original, cropped, dropped, attention maps
        create_and_save_subplot(data_for_visualization, config, batch_id_for_image_name, save_dir)

    return end_points




def load_nets_and_freeze_param(config, nets):
    if not config.single_network_model:
        if config.initial_model_kwargs["load_pretrained_base_model"]:
            nets.base_model.load_state_dict(torch.load(config.initial_model_kwargs["initial_model_base_path"]))
        if config.initial_model_kwargs["load_pretrained_tax_model"]:
            nets.emb_model.load_state_dict(torch.load(config.initial_model_kwargs["initial_model_tax_path"]))

        # Freeze certain ratio of taxonomy model parameters
        if config.freeze_weights_kwargs["freeze_weights"]:
            params_emb = list(nets.emb_model.parameters())

        # Calculate the number of parameters to freeze
        num_to_freeze = int(len(params_emb) * config.freeze_weights_kwargs["freeze_ratio_tax_model_parameters"])

        # Freeze the first percentage of layers
        for param in params_emb[:num_to_freeze]:
            param.requires_grad = False
        
    else:
        nets.base_model.load_state_dict(torch.load(config.initial_model_kwargs["initial_model_tax_path"]))
        params_base = list(nets.base_model.parameters())

        # Calculate the number of parameters to freeze
        num_to_freeze = int(len(params_base) * config.freeze_ratio_single_model)

        # Freeze the first percentage of layers
        for param in params_base[:num_to_freeze]:
            param.requires_grad = False

    return nets


