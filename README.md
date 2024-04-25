# Brain Tumor Detection 🧠

Welcome to the Brain Tumor Detection project! This web application uses deep learning to identify brain tumors from medical images, providing you with accurate diagnoses to assist in effective treatment. Let's dive in!

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation & Metrics](#evaluation--metrics)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Contact](#contact)

## Project Overview
Brain tumor detection is a complex challenge, but our project tackles it head-on! We're using a convolutional neural network (CNN) to classify brain tumors into four categories: Glioma, Meningioma, Pituitary, and No Tumor. The web application integrates the trained CNN model to make predictions on user-uploaded images.

## Dataset
The dataset includes 2,896 medical images across four classes:

- **Glioma Tumor**: 926 images
- **Meningioma Tumor**: 937 images
- **No Tumor**: 500 images
- **Pituitary Tumor**: 901 images

Organized as follows:


## Installation
1. **Clone the Repository**:  
    ```bash
    git clone https://github.com/your-username/brain-tumor-detection.git
    ```

2. **Install Dependencies**:  
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the Dataset**:  
    Ensure you have the dataset in the specified directory structure (`/dataset/Training` and `/dataset/Testing`).

## Getting Started
1. **Run the Flask Application**:  
    ```bash
    python Webapp.py
    ```

2. **Navigate to the Web App**:  
    Open your web browser and go to `http://127.0.0.1:5000/`.

3. **Upload Your Image**:  
    Upload an image of a brain scan through the web interface and receive predictions and confidence levels.

## Model Architecture
Our CNN architecture includes:

- **Convolutional Layers**: Feature extraction with ReLU activation.
- **Max-pooling Layers**: For dimensionality reduction.
- **Dropout Layers**: For regularization and preventing overfitting.
- **Fully Connected Layers**: For classification.
- **Output Layer**: Uses softmax activation for multi-class classification.

## Training the Model
Training involves:

1. **Data Loading**: Load images from specified training and testing directories.
2. **Data Preprocessing**: Normalization, resizing, and data augmentation (e.g., rotation, flipping, zooming).
3. **Data Splitting**: Training, validation, and test sets (e.g., 60% training, 20% validation, 20% testing).
4. **Model Compilation**: Use categorical cross-entropy loss and Adam optimizer.
5. **Model Training**: Train the model, monitoring validation loss and accuracy.
6. **Early Stopping**: Avoid overfitting by stopping training when validation loss plateaus.
7. **Model Saving**: Save trained model weights for future use and deployment.

## Evaluation & Metrics
We assess the model's performance using:

- **Accuracy**: How often the model is correct.
- **Precision, Recall, and F1-score**: For each class (Glioma, Meningioma, Pituitary, No Tumor) to understand performance across categories.

Consider using saliency maps, SHAP values, or other interpretability methods to explain the model's decisions.

## Deployment
Deploy the trained model as a web application using Flask. Options for deployment include local servers or cloud platforms (AWS, GCP). Consider using Docker for containerization.

## Contributing
We welcome contributions! If you encounter issues or have suggestions, please open an issue or submit a pull request.

## Contact
If you have questions or feedback, reach out to us:

- **Email**: aryansin2468@gmail.com
- **LinkedIn**: [Aryan Singh](https://www.linkedin.com/in/aryan-singh-162560260/)

Thank you for your interest in the project! Let's collaborate and make an impact in the medical domain!

