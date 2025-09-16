import os
import cv2

from vtcv_fine_grained_classification.src.utils.utils import to_numpy_img, process_attention_map, \
    create_attention_overlay, save_2by4_figures




def save_validation_subplot(
    original_img, 
    cropped_img, 
    dropped_img, 
    att_map_1, 
    att_map_2, 
    prop_top1, 
    prop_top2, 
    clss_top1,
    clss_top2,
    label, 
    index_in_batch,
    batch_id,
    pixel_max,
    pixel_min,
    save_root="./images_valid"
):
    """
    Saves a single figure containing:
      1) Original image
      2) Cropped image
      3) Dropped image
      4) Normalized Attention Map #1 overlay
      5) Normalized Attention Map #2 overlay
      6) Top 5% Attention Map #1
      7) Top 5% Attention Map #2

    in a folder structure:
      ./images_valid/<label>/valid_image_<index_in_batch>.jpg
    """
    ## Create a directory for the current label
    #label_dir = os.path.join(save_root, str(label))
    #os.makedirs(label_dir, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)

    original_img_np = to_numpy_img(original_img, pixel_max, pixel_min)
    cropped_img_np = to_numpy_img(cropped_img, pixel_max, pixel_min)
    dropped_img_np = to_numpy_img(dropped_img, pixel_max, pixel_min)



    # Get regular normalized maps
    att_map_1_norm = process_attention_map(att_map_1)
    att_map_2_norm = process_attention_map(att_map_2)
    
    # Get top 5% maps
    att_map_1_top5 = process_attention_map(att_map_1, top_percent=5)
    att_map_2_top5 = process_attention_map(att_map_2, top_percent=5)


    # Convert original image to BGR for the overlay step
    original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

    # Create overlay images
    overlay = [create_attention_overlay(original_img_bgr, att_map_1_norm, pixel_max, pixel_min)]
    overlay.append(create_attention_overlay(original_img_bgr, att_map_2_norm, pixel_max, pixel_min))
    overlay_top5 = [create_attention_overlay(original_img_bgr, att_map_1_top5, pixel_max, pixel_min, blend_ratio=0.85)]
    overlay_top5.append(create_attention_overlay(original_img_bgr, att_map_2_top5, pixel_max, pixel_min, blend_ratio=0.85))


    # Save figures
    #save_2by4_figures(original_img_np, cropped_img_np, dropped_img_np, overlay, overlay_top5, prop_top1, \
    #                  prop_top2, clss_top1, clss_top2, index_in_batch, batch_id, label_dir)    
    save_2by4_figures(original_img_np, cropped_img_np, dropped_img_np, overlay, overlay_top5, prop_top1, \
                      prop_top2, clss_top1, clss_top2, index_in_batch, batch_id, label, save_root)    



def create_and_save_subplot(val_data_for_visualization, config, batch_id_for_image_name, save_dir):
        # For the subplot, we need cropped and dropped images as well:
    # (Even if you don't use them in the final classification, we want them for visualization.)
    # # # # We assume end_points["prop"] has shape (B,2) = [top1, top2].

    num_img_in_cur_batch = val_data_for_visualization["im"].shape[0]
    for i in range(num_img_in_cur_batch):
        label_i = val_data_for_visualization["labels"][i].item()  # get the label of the ith sample
        prop_top1 = val_data_for_visualization["prop"][i][0].item()
        prop_top2 = val_data_for_visualization["prop"][i][1].item()
        clss_top1 = val_data_for_visualization['topk'][i][0].item()
        clss_top2 = val_data_for_visualization['topk'][i][1].item()

        # Single images for subplot
        img_original = val_data_for_visualization["im"][i]
        img_cropped  = val_data_for_visualization["im_crop"][i]
        img_dropped  = val_data_for_visualization["im_drop"][i]
        att_map_1    = val_data_for_visualization["att_maps"][i, 0]  # channel 1
        att_map_2    = val_data_for_visualization["att_maps"][i, 1]  # channel 2

        # Create and save the subplot
        save_validation_subplot(
            original_img   = img_original,
            cropped_img    = img_cropped,
            dropped_img    = img_dropped,
            att_map_1      = att_map_1,
            att_map_2      = att_map_2,
            prop_top1      = prop_top1,
            prop_top2      = prop_top2,
            clss_top1  = clss_top1 ,
            clss_top2  =clss_top2 ,
            
            label          = label_i,
            index_in_batch = i,
            batch_id = batch_id_for_image_name,
            pixel_max = config.max_pixel,
            pixel_min = config.min_pixel,
            save_root      = save_dir
        )
