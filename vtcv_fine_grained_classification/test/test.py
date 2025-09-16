import json
import torch
import time
from datetime import timedelta
from functools import *
from pathlib import Path

import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
from fastai.vision.all import *
from torch.utils.tensorboard import SummaryWriter

# ----------------------------------------------------------------------------------------
# Below is your custom imports:
# ----------------------------------------------------------------------------------------
from vtcv_fine_grained_classification.src.configs.train_config import TrainConfig
from vtcv_fine_grained_classification.src.configs.config import ExperimentationConfig
from vtcv_fine_grained_classification.src.configs.paths_config import PathsConfig
from vtcv_fine_grained_classification.src.models.region import *
from vtcv_fine_grained_classification.src.models.model import MainModel
from vtcv_fine_grained_classification.src.utils.utils import log_information
from vtcv_fine_grained_classification.src.models.build_model import build_model, define_losses
from vtcv_fine_grained_classification.test.metrics import normalize_confusion_matrix, calculate_metrics_with_confidence
from vtcv_fine_grained_classification.test.test_utils import create_pdf_report
# ----------------------------------------------------------------------------------------


class Tester(torch.nn.Module):
    def __init__(self, config, exp_dir, model):
        super().__init__()
        self.exp_dir = exp_dir
        self.config = config
        self.device = config.device
        self.model = model

        datasheet_dest_path = exp_dir / "full_data.csv"
        shutil.copy(config.datasheet_path, datasheet_dest_path)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def _load_checkpoint(self):
        print(f"Loading model from {self.exp_dir / 'model_last.pth'}")
        model = torch.load(self.exp_dir / f"model_last.pth")
        return model

    def test_(self, valid_dataloader):
        """
        Main entry point for testing. Computes test metrics, confusion matrices, logs results,
        and saves classification report as a PDF.
        """
        start_time = time.time()


        losses_val, losses_val_crop, losses_val_drop, loss_weight_val, val_acc, val_acc_crop, \
        val_acc_drop, val_acc_image_combined = self.test_epoch(valid_dataloader)
        elapsed = time.time() - start_time
        elapsed = str(timedelta(seconds=elapsed))[:-7]

        # Prepare logging output
        log_information(self.config, elapsed, (losses_val, ), \
                            (losses_val_crop, ), (losses_val_drop, ), \
                            (loss_weight_val, ), (val_acc, ), \
                            (val_acc_crop, ), (val_acc_drop, ), \
                            (val_acc_image_combined, ), self.optim.param_groups[0]['lr'])

        
        # Write out to TensorBoard (example metrics)
        writer = SummaryWriter(log_dir=f"runs/{self.config.exp_name}")
        writer.add_scalar("Loss/val_criterion_loss", losses_val["criterion_loss"])
        writer.add_scalar("Accuracy/val_accuracy_image", val_acc["accuracy_image"])
        writer.flush()
        writer.close()



    def test_epoch(self, dataloader):
        """
        Runs one pass over the test/validation set, calculates metrics, confusion matrix,
        classification report, and saves a PDF report.
        """
        total_criterion_loss = 0
        total_criterion_loss_crop = 0
        total_criterion_loss_drop = 0
        total_weighted_loss = 0

        correct_image = 0
        correct_image_crop = 0
        correct_image_drop = 0
        correct_image_combined = 0

        y_true = []
        y_pred = []

        for cur_batch , (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(self.device), labels.to(self.device)


            # Evaluate and comupute losses
            self.model.eval()
            # with torch.no_grad():
            criterion_loss, outputs = self.model(images, labels, cur_batch = cur_batch, mode="Test")
            
            loss_diversity =  outputs['loss_diversity']
            _, crop_loss, drop_loss, combined_loss = criterion_loss
            weighted_loss = combined_loss * self.config.weighted_loss_parameters["combined_loss_weight"] \
                            + loss_diversity * self.config.weighted_loss_parameters["diversity_loss_weight"]
            
            
            
            # Calculate total losses
            total_weighted_loss += weighted_loss.sum().item()
            total_criterion_loss += combined_loss.sum().item()
            total_criterion_loss_crop += crop_loss.sum().item()
            total_criterion_loss_drop += drop_loss.sum().item()


            # Predictions
            _, predicted = torch.max(outputs["logits"], 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            correct_image += (predicted == labels).sum().item()

            if self.config.weight_logits_crop != 0 and outputs["logits_crop"] is not None:
                _, predicted_crop = torch.max(outputs["logits_crop"], 1)
                correct_image_crop += (predicted_crop == labels).sum().item()

            if self.config.weight_logits_drop != 0 and outputs["logits_drop"] is not None:
                _, predicted_drop = torch.max(outputs["logits_drop"], 1)
                correct_image_drop += (predicted_drop == labels).sum().item()

            # Combined
            _, predicted_combined = torch.max(outputs["combined_logits"], 1)
            correct_image_combined += (predicted_combined == labels).sum().item()


        # Averages
        avg_loss_criterion = total_criterion_loss / len(dataloader.dataset)
        avg_loss_criterion_crop = total_criterion_loss_crop / len(dataloader.dataset)
        avg_loss_criterion_drop = total_criterion_loss_drop / len(dataloader.dataset)
        avg_weighted_loss = total_weighted_loss / len(dataloader.dataset)

        accuracy_image = 100.0 * correct_image / len(dataloader.dataset)
        accuracy_image_crop = 100.0 * correct_image_crop / len(dataloader.dataset)
        accuracy_image_drop = 100.0 * correct_image_drop / len(dataloader.dataset)
        accuracy_image_combined = 100.0 * correct_image_combined / len(dataloader.dataset)


        # Confusion matrix
        normalize_confusion_matrix(y_true, y_pred, self.exp_dir, norm="true", class_names=None)

        # Metrics
        # Use string class names or numeric indices
        class_names = [str(i) for i in range(self.config.num_classes)]
        metrics_df, macro_avg, micro_avg = calculate_metrics_with_confidence(
            y_true, y_pred, class_names
        )
        

        # Create PDF
        pdf_path = f"{self.exp_dir}/test_predictions/classification_metrics_report.pdf"
        create_pdf_report(metrics_df, macro_avg, micro_avg, pdf_path)

        return (
            {"criterion_loss": avg_loss_criterion},
            {"criterion_loss_crop": avg_loss_criterion_crop},
            {"criterion_loss_drop": avg_loss_criterion_drop},
            {"weighted_loss": avg_weighted_loss},
            {"accuracy_image": accuracy_image},
            {"accuracy_image_crop": accuracy_image_crop},
            {"accuracy_image_drop": accuracy_image_drop},
            {"accuracy_image_combined": accuracy_image_combined},
        )




def test(experiment_dir: str):
    """
    Primary test function that loads model, dataset, and runs Tester.
    """
    print("Testing the model****************************************************************")
    with open(experiment_dir + "/TrainConfig.json", "r") as json_file:
        settings_json = json.load(json_file)
        settings_json["mode"] = "test"
        config = ExperimentationConfig.parse_obj(settings_json)
        config_paths = PathsConfig(expconfig=config)
        train_config = TrainConfig(expconfig=config, paths_config=config_paths)
        exp_dir = Path(config.save_dir) / config.exp_name
        num_classes = train_config.get_num_classes(f"{exp_dir}/existing_data.csv")

    with (exp_dir / "TrainConfig.json").open("w") as frozen_settings_file:
        json.dump(config.dict(exclude_none=True), frozen_settings_file, indent=2)

    # Ensure dataset is processed
    train_config.save_processed_datasets()


    # Define network modules
    num_classes = train_config.get_num_classes(config.datasheet_path)
    device = config.device
    config.num_classes = num_classes
    nets = build_model(config, device, num_classes=num_classes)
    

    # Example freeze
    params_emb = list(nets.emb_model.parameters())
    num_to_freeze = int(len(params_emb) * 0.95)
    for param in params_emb[:num_to_freeze]:
        param.requires_grad = False


    # Create model
    criterion = train_config.get_loss(num_classes, device)
    losses = define_losses(criterion)
    model = MainModel(nets, losses, config)
    model = model.to(config.device)

    # Define tester
    tester = Tester(config, exp_dir, model)


    # Load pretrained model parameters
    print(f"{exp_dir}/model_best.pth")
    model.load_state_dict(torch.load(f"{exp_dir}/model_best.pth"))

    # Test dataset
    valid_set = train_config.get_dataset(mode="Test")
    valid_dataloader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Run tester
    tester.test_(valid_dataloader)


