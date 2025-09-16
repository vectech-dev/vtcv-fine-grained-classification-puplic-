import json
import cv2
import os
from os import  makedirs
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as path_effects


# Helper to convert [3,H,W] float tensor (0..1) to uint8 [H,W,3]
def to_numpy_img(tensor_img, pixel_max, pixel_min):
    np_img = tensor_img.cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, C]
    np_img = np.clip(np_img * pixel_max, pixel_min, pixel_max).astype(np.uint8)
    return np_img

# Process attention maps
def process_attention_map(att_map, top_percent=None):
    """Normalize attention map and optionally highlight top percent"""
    att_map_np = att_map.cpu().detach().numpy()
    
    # If we want only the top X% most salient regions
    if top_percent is not None:
        threshold = np.percentile(att_map_np, 100 - top_percent)
        att_map_np = np.where(att_map_np >= threshold, att_map_np, 0)
        # # Renormalize after thresholding
        # if att_map_norm.max() > 0:
        #     att_map_norm = att_map_norm / att_map_norm.max()
    
    return att_map_np

# ---- Overlay attention maps on the original image ----
def create_attention_overlay(image_np_bgr, att_map, pixel_max, pixel_min, blend_ratio=0.75):
    """
    image_np_bgr: [H,W,3] BGR image in uint8
    att_map: [H,W] in [0,1]
    blend_ratio: how much weight to give to heatmap (0-1)
    """
    heatmap = plt.get_cmap('jet')(att_map)[:, :, :3]  # shape [H, W, 3], ignoring alpha
    heatmap = np.clip(heatmap * pixel_max, pixel_min, pixel_max).astype(np.uint8)

    # Convert to BGR for consistent OpenCV blending
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    # Blend: original is (1-blend_ratio), heatmap is blend_ratio
    overlay = cv2.addWeighted(image_np_bgr, 1-blend_ratio, heatmap_bgr, blend_ratio, 0)
    # Convert back to RGB so matplotlib will show correct colors
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


#def save_2by4_figures(original_img_np, cropped_img_np, dropped_img_np, overlay, overlay_top5, \
#                      prop_top1, prop_top2, clss_top1, clss_top2, index_in_batch, batch_id, label_dir):
def save_2by4_figures(original_img_np, cropped_img_np, dropped_img_np, overlay, overlay_top5, \
                      prop_top1, prop_top2, clss_top1, clss_top2, index_in_batch, batch_id, label, save_root):

    # Create figure with adjusted spacing
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))  # Increased height for spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Increased horizontal/vertical spacing

    # First row: Original visualization
    axes[0,0].imshow(original_img_np)
    axes[0,1].imshow(cropped_img_np)
    axes[0,2].imshow(dropped_img_np)
    axes[0,3].axis('off')

    # Second row: Attention visualizations
    axes[1,0].imshow(overlay[0])
    axes[1,1].imshow(overlay[1])
    axes[1,2].imshow(overlay_top5[0])
    axes[1,3].imshow(overlay_top5[1])

    # Define titles with enhanced styling
    titles_config = [
        [f"Original (label = {label})", "Cropped", "Dropped", None],
        [
            f"Att Map 1\nTop1: {prop_top1:.2f} {clss_top1}",
            f"Att Map 2\nTop2: {prop_top2:.2f} {clss_top2}",
            "Top 5% Att Map 1",
            "Top 5% Att Map 2"
        ]
    ]

    # Apply styled titles
    for i in range(2):
        for j in range(4):
            if titles_config[i][j]:
                title = axes[i,j].set_title(
                    titles_config[i][j],
                    fontsize=16,          # Larger font size
                    color='white',         # White text
                    fontweight='bold',     # Bold text
                    pad=20                # Padding between title and image
                )
                # Add black outline for contrast
                title.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='black'),
                    path_effects.Normal()
                ])

    # Turn off axes and apply layout
    for ax in axes.flat:
        ax.axis('off')

    # Add extra padding around subplots
    plt.tight_layout(pad=4.0, w_pad=3.0, h_pad=3.0)

    # Save figure
    #print('Saving images')
    #output_file = os.path.join(label_dir, f"image_{batch_id}_{index_in_batch}.jpg")
    output_file = os.path.join(save_root, f"image_{batch_id}_{index_in_batch}.jpg")
    plt.savefig(output_file, bbox_inches='tight', dpi=150, facecolor='#202020')  # Dark background
    plt.close(fig)



def log_information(config, elapsed, loss, loss_crop, loss_drop, loss_weight, accuracy, \
                          accuracy_crop, accuracy_drop, accuracy_combined, lr, epoch = 0):
    if config.mode == "test":
        log = ""
        num_iter = 1
        md = ['test']
    else:
        log = f"Elapsed time [{elapsed}], Iteration [{epoch}/{config.epochs}], "
        num_iter = 2
        md = ['train', 'val']
    for i in range(num_iter):
        log += ' '.join([f"{md[i]}_{k}: [{v:.4f}]" for k, v in loss[i].items()])
        log += ' '.join([f"{md[i]}_{k}: [{v:.4f}]" for k, v in loss_crop[i].items()])
        log += ' '.join([f"{md[i]}_{k}: [{v:.4f}]" for k, v in loss_drop[i].items()])
        log += ' '.join([f"{md[i]}_{k}: [{v:.4f}]" for k, v in loss_weight[i].items()])
        log += ' '.join([f"{md[i]}_{k}: [{v:.4f}]" for k, v in accuracy[i].items()])
        log += ', '.join(f"{md[i]}_{k}: [{v:.4f}]" for k, v in accuracy_crop[i].items())
        log += ', '.join(f"{md[i]}_{k}: [{v:.4f}]" for k, v in accuracy_drop[i].items())
        log += ', '.join(f"{md[i]}_{k}: [{v:.4f}]" for k, v in accuracy_combined[i].items())
    log += f", lr: [{lr}]"
    print(log)
    

def save_images_and_att_maps(images, labels, outputs):
    data_dict = {}
    data_dict["im"] = images
    data_dict["labels"] = labels
    data_dict["im_crop"] = outputs['images_crop']
    data_dict["im_drop"] = outputs['images_drop']
    data_dict["att_maps"] = outputs['attention_map']
    data_dict["prop"] = outputs['prop']
    data_dict["topk"] = outputs['topk']

    return data_dict



def save_time_tracking(durations_over_all_batches, filepath, cur_epoch):
    avg_duration = {key: value for key, value in durations_over_all_batches[0].items()}
    for i in range(1, len(durations_over_all_batches)):
        for key, value in durations_over_all_batches[i].items():
            avg_duration[key] += value
    avg_duration = {key: value / len(durations_over_all_batches) for key, value in avg_duration.items()}
    if cur_epoch == 1:
        with (filepath).open("w") as json_file:
            json.dump([avg_duration], json_file, indent = 4)
    else:
        append_to_json_file(filepath, avg_duration)


def append_to_json_file(filename, new_entry):
    try:
        # Read the existing data from the file
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, initialize data as an empty list
        data = []
    except json.JSONDecodeError:
        # Handle cases where the file might be empty or malformed
        print(f"Warning: JSON file '{filename}' is empty or malformed. Initializing as empty list.")
        data = []

    # Ensure data is a list before appending
    if not isinstance(data, list):
        print(f"Error: JSON file '{filename}' does not contain a list. Cannot append.")
        return

    # Append the new entry to the list
    data.append(new_entry)

    # Write the updated data back to the file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4) # indent for pretty printing

class SimpleLogger:
    def __init__(self, name):
        self.name = name

    def log(self, data):
        print(f"{self.name} - {data}")


def make_dirs(dirs):
    for dir_ in dirs:
        makedirs(dir_, exist_ok=True)


def he_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    return m