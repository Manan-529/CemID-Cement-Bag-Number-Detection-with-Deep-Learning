# CemID-Cement-Bag-Number-Detection-with-Deep-Learning

## Overview
CemID: Deep Learning-Powered Real-Time Cement Bag Number Detection is a sophisticated system designed to identify week numbers on cement bags in real time as they move along a conveyor belt. Leveraging advanced deep learning techniques, the system ensures accurate detection of week numbers, enhancing efficiency and quality control in cement manufacturing processes. By integrating TensorFlow for model training and OpenCV for image processing, CemID achieves high accuracy even in challenging environments with varying lighting conditions and bag orientations. Its pause mechanism halts the conveyor belt upon detecting a bag without a week number, preventing defective products from progressing further in the manufacturing process.



## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup and Installation](#setup-and-installation)
3. [Training the Model](#training-the-model)
4. [Running Predictions](#running-predictions)
5. [Output](#output)
6. [Skills Demonstrated](#skills-demonstrated)

## Project Structure
- **data/**: Contains annotated images, original images, and video files.
- **scripts/**: Contains training and prediction scripts.
- **output/**: Contains example video showing the detection in action.
- **requirements.txt**: Lists the Python dependencies.
- **README.md**: Project documentation.

## Setup and Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/Manan-529/CemID-Cement-Bag-Number-Detection-with-Deep-Learning.git
    cd CemID-Cement-Bag-Number-Detection-with-Deep-Learning
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Place your annotated images, original images, and video files in the respective folders under `data/`.

## Training the Model
1. Ensure your images are in the correct folders under `data/`.
2. Run the training script:
    ```sh
    python scripts/train.py
    ```

## Running Predictions
1. Ensure your video files are in the `data/vdos/` folder.
2. Run the prediction script:
    ```sh
    python scripts/predict.py
    ```




## Output
- Example of the model in action:


https://user-images.githubusercontent.com/52171362/157252684-bd0c3e4b-063f-4b20-9bcb-58eccc6143da.mp4


## Skills Demonstrated
- **Computer Vision**: Implemented using OpenCV for video processing and image manipulation.
- **Deep Learning**: Utilized TensorFlow for building and training the SegNet model.
- **Data Preprocessing**: Included steps for resizing images, data normalization, and creating datasets.
- **Model Training**: Demonstrated ability to train a deep learning model with validation and saving checkpoints.
- **Python Programming**: Strong coding skills are shown in the project scripts and organization.
