from setuptools import setup, find_packages

# Load the README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="yolo_adversarial_example_generation",
    version="0.1.0",
    author="J. Reber",
    description="Adversarial Example Generation for YOLO Object Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/j-reber/yolo_adversarial_example_generation",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "torch>=1.6.0",
        "matplotlib>=3.2.0",
        "Pillow>=7.0.0",
        "opencv-python>=4.2.0.34",
        "scipy>=1.4.1",
        "tqdm>=4.42.0",
        # Add any other dependencies that are listed in the repository's requirements.txt or are needed by the package
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
