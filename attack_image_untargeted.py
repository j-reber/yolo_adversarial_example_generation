import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
import torch
from torchvision import transforms
from types import SimpleNamespace
import argparse
import cv2
from attack_image_targeted import BaseAttacker

class UntargetedAttacker(BaseAttacker):
    def __init__(self, model_path, gamma=0.01, max_iter=60):
        super().__init__(model_path, gamma, max_iter)

    def attack_image(self, input_path):
        im = cv2.imread(input_path)
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640)),
        ])

        img_tensor = t(im)[None, :, :, :].float()
        img_tensor = torch.clamp(img_tensor, 0., 1.)
        # Record gradients for image
        img_tensor.requires_grad = True

        # Setup the models by inferring them once
        initial_results = self.model.predict(img_tensor)[0]
        predictor = getattr(self.model, 'predictor', None)
        ultra_model = getattr(predictor, 'model', None)
        self.torch_model = getattr(ultra_model, 'model', None)

        classes = initial_results.boxes.cls
        boxes = initial_results.boxes.xyxyn
        perturbation = torch.zeros(img_tensor.shape, dtype=torch.float32).to(self.device)

        target_labels = {
            "batch_idx": torch.arange(boxes.shape[0]).to(self.device),  # Generating an index tensor
            "cls": classes.view(-1, 1),  # Reshaping to match expected dimensions
            "bboxes": boxes.to(self.device),
        }
        loss_fn = v8DetectionLoss(self.torch_model)
        foo = SimpleNamespace(box=1, cls=1, dfl=1)  # Reusable code much?
        loss_fn.hyp = foo

        print("Starting attack...")
        for _ in tqdm(range(self.max_iter)):  # Start DAG here
            self.torch_model.train()
            res = self.torch_model(img_tensor.to(self.device))
            target_loss = loss_fn(res, target_labels)

            # Make every target incorrectly predicted as the adversarial label
            total_loss = target_loss[0]
            # Backprop and compute gradient wrt image
            total_loss.backward()
            image_grad = img_tensor.grad.detach()

            # Apply perturbation on image
            with (torch.no_grad()):
                # Normalize grad
                perturbation = (self.gamma / image_grad.norm(float("inf"))) * image_grad
                perturbation += perturbation
                img_tensor += perturbation

            # Zero Gradients (really necessary?)
            image_grad.zero_()
            self.torch_model.zero_grad()

        # Normalize img tensor
        # img_tensor_min = img_tensor.min()
        # img_tensor_max = img_tensor.max()
        # img_tensor = (img_tensor - img_tensor_min) / (img_tensor_max - img_tensor_min)
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return img_tensor, perturbation



def main(model_path, image_path, output_path, gamma, max_iter, save_perturbation, save_adv_image, save_label):
    attacker = UntargetedAttacker(model_path, gamma, max_iter)
    im, per = attacker.attack_image(image_path)
    attacker.save_result(im, output_path)
    attacker.save_perturbation(per, save_perturbation)
    attacker.save_adv_image(im, save_adv_image)
    attacker.save_label(im, save_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Adversarial Examples from a given image')
    parser.add_argument('--model', type=str, required=False, default="yolov8n.pt",
                        help='Path to the trained YOLOv8 model file (e.g., yolov8l.pt or /path/to/best.pt).')
    parser.add_argument('--image_path', type=str, required=False, default='test_data/johannes_lukas.jpg',
                        help='Path to the image to create adversarial examples from.')
    parser.add_argument('--output_path', type=str, required=False, default=None,
                        help='Path to the save location of the image. Do not specify for not saving.')
    parser.add_argument('--gamma', type=float, required=False, default=0.01,
                        help='Set the hyperparameter gamma for gradient normalization.')
    parser.add_argument('--max_iter', type=int, required=False, default=60,
                        help='Set the maximum iterations for the DAG algorithm.')
    parser.add_argument('--save_perturbation', type=str, required=False, default=None,
                        help='Path to the save location of the perturbation. Do not specify for not saving.')
    parser.add_argument('--save_adv_image', type=str, required=False, default=None,
                        help='Path to the save location of the adversarial image. Do not specify for not saving.')
    parser.add_argument('--save_label', type=str, required=False, default=None,
                        help='Path to the save location of the Label. Do not specify for not saving.')
    args = parser.parse_args()
    main(args.model, args.image_path, args.output_path, args.gamma, args.max_iter, args.save_perturbation,
         args.save_adv_image, args.save_label)
