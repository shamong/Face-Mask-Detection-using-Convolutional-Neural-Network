# Face-Mask-Detection-using-Convolutional-Neural-Network
This project trains a CNN to detect face masks in images. Using a labeled Kaggle dataset, the CNN learns to classify masked vs. unmasked faces. It achieves an accuracy of 91.8% on unseen data, suggesting potential for real-world public health or security applications.


## Overview

The Face Mask Detection project utilizes Convolutional Neural Networks (CNN) to identify individuals wearing face masks within images. The project involves data preprocessing, model development, training, and real-time predictions. Additionally, it proposes an AI-based detection system to identify social distancing violations.

## Source

Kaggle - https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

WHO COVID-19 dashboard - https://data.who.int/dashboards/covid19/cases?n=c

Findings - https://www.nebraskamed.com/COVID

## Research Objective

The project aims to develop a robust face mask detection system using CNN by accurately classifying images into 'with mask' and 'without mask' categories. This system contributes to safety measures in various environments and assists in curbing the transmission of COVID-19.

## Keyword Understanding

#Central Concepts

1. Computer Vision: Processing visual data to enable machines to interpret images.

2. Deep Learning: Neural networks learning intricate patterns from data, suitable for tasks like image recognition.

3. Image Classification: Assigning labels to images based on their content.

4. Binary Classification (Mask vs. No Mask): Distinguishing between images with and without masks.

## Key Technical Frameworks

1. CNN: Neural networks for analyzing visual data.

2. Ensemble Methods: Combining multiple models to enhance accuracy.

3. Techniques: Advanced methodologies to improve model performance.

## Image Handling Tools

1. NumPy: Used for numerical operations and manipulation of arrays.

2. OpenCV: A library for computer vision tasks.

3. Matplotlib: A plotting library for visualizing images and data.

4. PIL (Python Imaging Library): A library for opening and manipulating image files.

## CNN Architecture

1. Layers: Components like Conv2D for convolution and Dense layers for classification.

2. Activation Functions: Functions like ReLU introduce non-linearity in the network.

3. Regularization: Techniques like Dropout prevent overfitting.

## Training Methodology

1. Dataset Splitting: Partitioning data into training and validation sets.

2. Data Scaling: Normalizing data to enhance model training.

3. Model Compilation: Configuring the model with an optimizer and loss function.

4. Training Epochs: Iteratively training the model for a specified number of epochs.

5. Model Saving and Evaluation: Saving and evaluating the trained model's performance.

## Methodology

1. Data Collection: Obtaining the dataset from Kaggle.

2. Data Organization and Inspection: Verifying dataset integrity.

3. Image Processing and Preparation: Standardizing and preprocessing images.

4. Construction of CNN: Assembling the CNN architecture.

5. Model Training and Evaluation: Training the model and evaluating its performance.

6. Prediction and Deployment: Developing a predictive system for real-world applications.

## Results

The CNN model achieves high accuracy in identifying mask-wearing individuals, showcasing its viability for deployment in various scenarios.

## Limitations

Potential limitations include variations in real-world scenarios and biases inherent in the dataset.

## Conclusion

The Face Mask Detection project successfully develops a robust model for identifying mask presence in images, contributing to safety measures in various environments.
