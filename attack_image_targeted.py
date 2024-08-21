import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
import torch
from torchvision import transforms
from types import SimpleNamespace
import argparse
import cv2
from abc import ABC, abstractmethod

class BaseAttacker(ABC):
    def __init__(self, model_path, gamma=0.01, max_iter=60):
        self.model = YOLO(model_path)
        self.torch_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.max_iter = max_iter
        self.last_result = None
    @abstractmethod
    def attack_image(self, input_path):
        pass
    def save_perturbation(self, data, output_path):
        if output_path is not None:
            self.torch_model.eval()
            perturbation_img = data.numpy()[0]
            perturbation_img = (perturbation_img * 255).astype(np.uint8)
            perturbation_img = np.transpose(perturbation_img, (1, 2, 0))
            cv2.imwrite(output_path, perturbation_img)

    def save_result(self, data, output_path):
        if output_path is not None:
            self.torch_model.eval()
            final_res = self.model.predict(data)
            final_res[0].save(filename=output_path)

    def save_adv_image(self, data, output_path):
        if output_path is not None:
            self.torch_model.eval()
            adv_img =data.detach().numpy()[0]
            adv_img = (adv_img * 255).astype(np.uint8)
            adv_img = np.transpose(adv_img, (1, 2, 0))
            cv2.imwrite(output_path, adv_img)
    def save_label(self, data, output_path):
        """
        Save the results as YOLO format labels.
        :param data: Perturbed normalized image
        :param output_path: (String)
        :return: None
        """
        if output_path is not None:
            self.torch_model.eval()
            final_res = self.model.predict(data)[0]
            with open(output_path, 'w') as f:
                for i in range(final_res.boxes.shape[0]):
                    class_id = int(final_res.boxes.cls[i])
                    x_center, y_center, width, height = final_res.boxes.xywhn[i]
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


class TargetedAttacker(BaseAttacker):
    def __init__(self, model_path, gamma=0.01, max_iter=60, seed=42):
        super().__init__(model_path, gamma, max_iter)
        self.seed = seed

    def attack_image(self, input_path):
        im = cv2.imread(input_path)
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640)),
        ])
        img_tensor = t(im)[None, :, :, :].float()
        img_tensor = torch.clamp(img_tensor, 0., 1.)
        img_tensor.requires_grad = True

        initial_results = self.model.predict(img_tensor)[0]
        number_of_instances = initial_results.boxes.cls.shape[0]
        classes = initial_results.boxes.cls
        boxes = initial_results.boxes.xyxyn
        nc = len(initial_results.names)

        predictor = getattr(self.model, 'predictor', None)
        ultra_model = getattr(predictor, 'model', None)
        self.torch_model = getattr(ultra_model, 'model', None)

        # Setup Perturbation
        perturbation = torch.zeros(img_tensor.shape, dtype=torch.float32).to(self.device)

        # Setup target_labels
        target_labels = {
            "batch_idx": torch.arange(number_of_instances).to(self.device),  # Generating an index tensor
            "cls": classes.view(-1, 1),  # Reshaping to match expected dimensions
            "bboxes": boxes
        }

        # Setup adversarial labels
        torch.manual_seed(self.seed)
        adv_cls_labels = torch.randint(0, nc, (number_of_instances,)).to(self.device)
        for i in range(adv_cls_labels.shape[0]):
            print(initial_results.names[adv_cls_labels[i].item()])
        adv_labels = {
            "batch_idx": torch.arange(number_of_instances).to(self.device),  # Generating an index tensor
            "cls": adv_cls_labels.view(-1, 1),  # Reshaping to match expected dimensions
            "bboxes": boxes
        }
        # Setup custom yolo loss function
        loss_fn = v8DetectionLoss(self.torch_model)
        foo = SimpleNamespace(box=1, cls=1, dfl=1)  # Reusable code much?
        loss_fn.hyp = foo

        print("Starting attack...")
        for _ in tqdm(range(self.max_iter)):  # Start DAG here
            self.torch_model.train()
            res = self.torch_model(img_tensor.to(self.device))
            target_loss = loss_fn(res, target_labels)
            adv_loss = loss_fn(res, adv_labels)

            # Make every target incorrectly predicted as the adversarial label
            total_loss = target_loss[0] - adv_loss[0]

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

def main(model_path, image_path, output_path, gamma, max_iter, save_perturbation, save_adv_image, seed, save_label):
    attacker = TargetedAttacker(model_path, gamma, max_iter, seed)
    im, per = attacker.attack_image(image_path)
    print(output_path)
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
                        help='Path to the save location of the image. None for not saving.')
    parser.add_argument('--gamma', type=float, required=False, default=0.01,
                        help='Set the hyperparameter gamma for gradient normalization.')
    parser.add_argument('--max_iter', type=int, required=False, default=60,
                        help='Set the maximum iterations for the DAG algorithm.')
    parser.add_argument('--seed', type=int, required=False, default=42,
                        help='Set a seed to choose the adversarial labels.')
    parser.add_argument('--save_perturbation', type=str, required=False, default=None,
                        help='Path to the save location of the perturbation, None for not saving.')
    parser.add_argument('--save_adv_image', type=str, required=False, default=None,
                        help='Path to the save location of the adversarial image, None for not saving.')
    parser.add_argument('--save_label', type=str, required=False, default=None,
                        help='Path to the save location of the Label, None for not saving.')
    args = parser.parse_args()
    main(args.model, args.image_path, args.output_path, args.gamma, args.max_iter, args.save_perturbation,
         args.save_adv_image, args.seed, args.save_label)
