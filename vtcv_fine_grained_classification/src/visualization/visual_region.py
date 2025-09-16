
import torch

def apply_col_outside_box(image: torch.Tensor, x1: int, y1: int, x2: int, y2: int, background_RGB) -> torch.Tensor:
    _, channels, height, width = image.shape
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)

    mask = torch.zeros_like(image, dtype=torch.bool)
    mask[:, :, y1:y2, x1:x2] = 1  # Correct indexing for 4D tensor

    background = torch.tensor(background_RGB, dtype=torch.float32).div(255).view(1, 3, 1, 1)
    if channels != 3:
        raise ValueError(f"Expected image with 3 channels (RGB), but got {channels} channels.")

    # Move background to the same device as the image
    background = background.to(image.device)
    image = torch.where(mask, image, background)
    return image.squeeze(0)  # Remove batch dimension


def crop_images(
    images: torch.Tensor,
    bboxes: torch.Tensor,
    config,
) -> torch.Tensor:
    cropped_images = []
    for i in range(len(images)):
        bbox = bboxes[i]
        # Convert bbox from normalized coordinates [0,1] to image coordinates
        # Now assumes bbox is in the order [xmin, ymin, xmax, ymax]
        
        start_x = int(bbox[0].item() * config.imsize)
        start_y = int(bbox[1].item() * config.imsize)
        end_x = int(bbox[2].item() * config.imsize)
        end_y = int(bbox[3].item() * config.imsize)
        
        # Apply color outside the bounding box
        new_image = apply_col_outside_box(images[i].unsqueeze(0), start_x, start_y, end_x, end_y, config.backgournd_RGB_cropped)
        
        cropped_images.append(new_image)
        
    return torch.stack(cropped_images)










