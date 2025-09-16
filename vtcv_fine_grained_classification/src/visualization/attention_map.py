import torch
import torch.nn.functional as F




class GradCAM:
    def __init__(self, base_model, target_layer):
        """
        :param base_model: The updated Xception_V2_1 model
        :param target_layer: The layer to compute Grad-CAM for (base_model.conv)
        """
        self.base_model = base_model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        #print(f"out[0].shape = {out[0].shape}")
        self.activations = out

    def backward_hook(self, module, grad_in, grad_out):
        #print(f"grad_out[0].shape = {grad_out[0].shape}")
        self.gradients = grad_out[0]

    def __call__(self, images, labels, epsilon = 1e-7):
        images.requires_grad_()
        
        self.base_model.eval()   # Added later

        # 1) Forward
        logits, end_points = self.base_model(images)  # shape: (B, num_classes)

        # 2) Backprop for each image i, target_class = labels[i]
        batch_inds = torch.arange(images.size(0))
        chosen_logits = logits[batch_inds, labels]  # shape: (B,)
        
        self.base_model.zero_grad()     # Added later
        
        chosen_logits.sum().backward(retain_graph=True)
        
        # 3) Grad-CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM did not capture gradients/activations!")
        #print(f"self.gradients.shape: {self.gradients.shape}")

        weights = torch.mean(self.gradients, dim=(2,3), keepdim=True)  # (B, 768, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1)             # (B, 17, 17)

        cam = F.relu(cam)
        cam_min, _ = cam.view(cam.size(0), -1).min(dim=1, keepdim=True)
        cam_max, _ = cam.view(cam.size(0), -1).max(dim=1, keepdim=True)
        cam = (cam - cam_min.unsqueeze(-1)) / (cam_max.unsqueeze(-1) + epsilon)  # 0~1
        return cam
    
# **Function to generate Grad-CAM images**
def generate_gradcam(images, labels, model):
    
    #Computes Grad-CAM images for each input image.
    
    #:param images: Batch of input images (B, C, H, W)
    #:param labels: Ground truth labels (B,)
    #:param nets: Model dictionary containing base_model and emb_model

    # Ensure the correct conv layer exists
    target_layer = model.block11
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    cam_maps = grad_cam(images, labels)  # Compute Grad-CAM for each image in batch

    cam_maps = cam_maps.view(cam_maps.shape[0], 1, cam_maps.shape[1], cam_maps.shape[2])
    return  cam_maps #F.interpolate(cam_maps, size=(299, 299), mode='bicubic', align_corners=True, antialias=True) 


