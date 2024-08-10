# Adversarial Example Generation for YOLOv8 Models
This repository implements the
[Dense Adversary Generation Algorithm (Xie et al., ICCV 2017)](https://arxiv.org/abs/1703.08603).
## Installation
To make the repository work follow these steps.
```
git clone https://github.com/j-reber/yolo_adversarial_example_generation.git
```
```
python -m venv venv \\
source venv/bin/activate
```
```
pip install requirements.txt
```
## Usage
```
 python3 attack_image_non_targeted.py --max_iter 15  --save_perturbation test_data/per.jpg
```
For a list of all available options for the script call: 
```
  python3 attack_image_non_targeted.py --help
```
## Pull Request and Enhancements
Please note that this repository now only works for YOLOv8 Detection models. Feel free to contribute.