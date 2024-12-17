# Footballer Image Classification System

This project implements a **Football Players Image Classification System** using **Convolutional Neural Networks (CNN)** with tools like OpenCV, NumPy, Matplotlib, Pandas, and Scikit-Learn.

## Table of Contents
1. [Overview](#overview)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Model Workflow](#model-workflow)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Acknowledgments](#acknowledgments)

---

## Overview
This system classifies images of football players using CNN models built with Python. The project processes and trains on labeled image datasets and achieves effective predictions for player recognition tasks.

---

## Technologies Used
- **Python 3.x**
- **OpenCV** (`cv2`) - Image loading & preprocessing
- **NumPy** - Numerical operations
- **Pandas** - Data handling
- **Matplotlib** - Visualization
- **Scikit-Learn** - Classification, preprocessing
- **Keras/TensorFlow** - CNN model building & training

---

## Dataset
The dataset consists of labeled images of football players. Each folder within the `image_dataset` directory corresponds to a player class.

---

## Model Workflow
1. **Data Loading**: Images loaded using OpenCV.
2. **Preprocessing**:
   - Resizing images to fixed dimensions.
   - Converting to grayscale or RGB as required.
3. **Model Training**:
   - CNN architecture is defined using Keras.
   - Dataset is split into training and validation sets.
   - The model is trained and evaluated.
4. **Evaluation**: Performance is visualized using accuracy and loss graphs.
5. **Classification**: Input images are classified into football player classes.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dunchutt/FootballerClassifier.git
   cd FootballerClassifier
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Place your labeled image dataset in the image_dataset folder.

## Usage

-**Run the training script:**
python train_model.py
-**Evaluate the model:**
python evaluate_model.py
-**Use the classifier:**
python classify_image.py --image path/to/image.jpg

## Results

The system achieves high accuracy in recognizing player images.

## Acknowledgments

Special thanks to the open-source ML and Python community for libraries like OpenCV, Scikit-Learn, and TensorFlow/Keras.
