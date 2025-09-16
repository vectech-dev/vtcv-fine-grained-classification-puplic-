import torch


def attention_crop(
    attention_maps: torch.Tensor,
    threshold_ratio: float,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:

    batch_size, _ , height, width = attention_maps.shape
    bboxes = []

    for i in range(batch_size):
        attention_map = attention_maps[i]
        # Sum across channels to get a single attention map
        part_sums = attention_map.sum(dim=0).to(device)  # Shape: [height, width]

        # Apply threshold
        gamma_min = threshold_ratio * part_sums.max()
        thresholded = (part_sums >= gamma_min).float() * part_sums

        # Find non-zero indices
        itemindex = torch.nonzero(thresholded, as_tuple=False)

        if itemindex.numel() > 0:
            # Extract bounding box coordinates
            ymin = itemindex[:, 0].min().item() / height
            ymax = itemindex[:, 0].max().item() / height
            xmin = itemindex[:, 1].min().item() / width
            xmax = itemindex[:, 1].max().item() / width
            bbox = torch.tensor([xmin, ymin, xmax, ymax], device=device)  # [xmin, ymin, xmax, ymax]
        else:
            bbox = torch.tensor([0, 0, 1, 1], device=device)  # Full image as default

        bboxes.append(bbox)

    bboxes = torch.stack(bboxes)
    return bboxes


def attention_drop(images, attention_maps, d_phi, similarity_threshold=-1):
    batch_size, channels, height, width = attention_maps.shape
    final_masks = []
    for i in range(batch_size):
        attention_map = attention_maps[i]
        max_attention = torch.max(attention_map, dim=0)[0]
        #print(f"attention_map.shape = {attention_map.shape}")
        #print(f"max_attention.shape = {max_attention.shape}")
        global_threshold = d_phi * torch.max(max_attention)
        common_mask = torch.zeros((height,width), dtype=torch.bool, device=attention_maps.device)
        for ch in range(channels):
            common_mask |= ((channels * similarity_threshold) < attention_map[ch]) & (attention_map[ch] <= global_threshold)
        final_masks.append(common_mask.unsqueeze(0))
    
    final_masks_tensor = torch.stack(final_masks, dim=0).to(torch.float)

    
    return images * final_masks_tensor.to(images.device)
