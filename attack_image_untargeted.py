import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
import torch
from torchvision import transforms
from types import SimpleNamespace
import argparse
import cv2


def main(model_path, image_path, output_path, gamma, max_iter, save_perturbation, save_adv_image, seed):
    model = YOLO(model_path)
    im = cv2.imread(image_path)
    # im = np.zeros((640, 640, 3), dtype=np.uint8)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
    ])

    img_tensor = t(im)[None, :, :, :].float()
    img_tensor = torch.clamp(img_tensor, 0., 1.)
    # Record gradients for image
    img_tensor.requires_grad = True

    # Setup CUDA if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Setup the models by inferring them once
    initial_results = model.predict(img_tensor)[0]
    predictor = getattr(model, 'predictor', None)
    ultra_model = getattr(predictor, 'model', None)
    torch_model = getattr(ultra_model, 'model', None)

    number_of_instances = initial_results.boxes.cls.shape[0]
    # classes = initial_results.boxes.cls
    boxes = initial_results.boxes.xyxyn
    # boxes_adv = torch.tensor([[0., 0., 1., 1.]])
    # number_of_adv_instances = boxes_adv.shape[0]
    nc = len(initial_results.names)

    # Setup Perturbation
    perturbation = torch.zeros(img_tensor.shape, dtype=torch.float32).to(device)

    # Setup target_labels
    classes = torch.randint(0, nc, (boxes.shape[0],)).to(device)
    target_labels = {
        "batch_idx": torch.arange(boxes.shape[0]).to(device),  # Generating an index tensor
        "cls": classes.view(-1, 1),  # Reshaping to match expected dimensions
        "bboxes": boxes.to(device),
    }
    #
    # # Setup adversarial labels
    # torch.manual_seed(seed)
    # adv_cls_labels = torch.randint(0, nc, (number_of_adv_instances,)).to(device)
    # for i in range(adv_cls_labels.shape[0]):
    #     print(initial_results.names[adv_cls_labels[i].item()])
    #
    # adv_labels = {
    #     "batch_idx": torch.arange(number_of_adv_instances).to(device),  # Generating an index tensor
    #     "cls": adv_cls_labels.view(-1, 1),  # Reshaping to match expected dimensions
    #     "bboxes": boxes_adv.to(device)
    # }
    # Setup custom yolo loss function
    loss_fn = v8DetectionLoss(torch_model)
    foo = SimpleNamespace(box=1, cls=1, dfl=1)  # Reusable code much?
    loss_fn.hyp = foo

    print("Starting attack...")
    for _ in tqdm(range(max_iter)):  # Start DAG here
        torch_model.train()
        res = torch_model(img_tensor.to(device))
        target_loss = loss_fn(res, target_labels)
        # target_loss = torch.zeros(1).to(device)
        # adv_loss = loss_fn(res, adv_labels)
        # print(target_loss[0], adv_loss[0])

        # Make every target incorrectly predicted as the adversarial label
        # total_loss = target_loss[0] - adv_loss[0]
        total_loss = target_loss[0]
        # print(adv_loss[0].item())
        # Backprop and compute gradient wrt image
        total_loss.backward()
        image_grad = img_tensor.grad.detach()

        # Apply perturbation on image
        with (torch.no_grad()):
            # Normalize grad
            perturbation = (gamma / image_grad.norm(float("inf"))) * image_grad
            perturbation += perturbation
            img_tensor += 0.01 * perturbation

        # Zero Gradients (really necessary?)
        image_grad.zero_()
        torch_model.zero_grad()

    # Normalize img tensor
    # img_tensor_min = img_tensor.min()
    # img_tensor_max = img_tensor.max()
    # img_tensor = (img_tensor - img_tensor_min) / (img_tensor_max - img_tensor_min)
    img_tensor = torch.clamp(img_tensor, 0, 1)

    torch_model.eval()
    if save_adv_image is not None:
        adv_img = img_tensor.detach().numpy()[0]
        adv_img = (adv_img * 255).astype(np.uint8)
        adv_img = np.transpose(adv_img, (1, 2, 0))
        cv2.imwrite(save_adv_image, adv_img)
    if save_perturbation is not None:
        perturbation_img = perturbation.numpy()[0]
        perturbation_img = (perturbation_img * 255).astype(np.uint8)
        perturbation_img = np.transpose(perturbation_img, (1, 2, 0))
        cv2.imwrite(save_perturbation, perturbation_img)
    if output_path is not None:
        final_res = model.predict(img_tensor)
        final_res[0].save(filename=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Adversarial Examples from a given image')
    parser.add_argument('--model', type=str, required=False, default="yolov8n.pt",
                        help='Path to the trained YOLOv8 model file (e.g., yolov8l.pt or /path/to/best.pt).')
    parser.add_argument('--image_path', type=str, required=False, default='test_data/johannes_lukas.jpg',
                        help='Path to the image to create adversarial examples from.')
    parser.add_argument('--output_path', type=str, required=False, default='test_data/result_untargeted.jpg',
                        help='Path to the save location of the image. None for not saving.')
    parser.add_argument('--gamma', type=float, required=False, default=0.5,
                        help='Set the hyperparameter gamma for gradient normalization.')
    parser.add_argument('--max_iter', type=int, required=False, default=150,
                        help='Set the maximum iterations for the DAG algorithm.')
    parser.add_argument('--seed', type=int, required=False, default=42,
                        help='Set a seed to choose the adversarial labels.')
    parser.add_argument('--save_perturbation', type=str, required=False, default=None,
                        help='Path to the save location of the perturbation, None for not saving.')
    parser.add_argument('--save_adv_image', type=str, required=False, default=None,
                        help='Path to the save location of the adversarial image, None for not saving.')
    args = parser.parse_args()
    main(args.model, args.image_path, args.output_path, args.gamma, args.max_iter, args.save_perturbation,
         args.save_adv_image, args.seed)
