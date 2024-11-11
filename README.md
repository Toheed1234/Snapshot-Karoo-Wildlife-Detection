# Snapshot Karoo Wildlife Detection
This project uses the Snapshot Karoo dataset to train and test a machine learning model for detecting and classifying animal species from camera trap images. The task is broken down into several stages, including data loading, cleaning, preprocessing, model training, and result analysis.

Dataset
The Snapshot Karoo dataset contains 14,889 sequences of camera trap images, totaling 38,074 images. These images are captured from different camera traps in the Karoo National Park, a wildlife reserve located in South Africa. The dataset is categorized into 38 species, with 83.02% of the images labeled as empty (no animals detected).

You can access the dataset here.

Tasks Overview
The project follows these key tasks:

Task 1: Data Selection and Loading
Goal:
Load the selected images and annotations from the Snapshot Karoo dataset into the model.

Description:

Camera Trap Folders: The data is divided into multiple folders corresponding to different camera traps, such as A01, A02, D01, etc.
Dataset Size: The initial dataset is 2.9GB with 2,799 images. The final dataset size has been reduced to approximately 1,000 images.
Task 2: Visualization
Goal:
Understand the distribution of species and the camera trap data.

Description:

Species Breakdown: Visualize the number of images for each species.
Empty Images: Visualize the count of images without animals.
Camera Trap Distribution: Show how many images were captured by each camera trap.
Species by Camera Trap: Visualize the species captured by each camera trap.
Task 3: Data Cleaning
Goal:
Clean and prepare the dataset for better classification performance.

Description:

Feature Selection: Optimize the feature space by selecting relevant features.

Removing Human Images: Remove any images with humans detected using MegaDetector.

MegaDetector:
A pre-trained model to detect animals, people, and vehicles in camera trap images. The model will remove any images without animals or those with confidence scores below 0.2.

Tasks:

Remove Low Confidence Detections: Filter images based on detection confidence (e.g., using a threshold of 0.2).
Cropping: Crop images to the bounding boxes containing detected animals.
Padding and Resizing: Ensure images are square and resized to 244x244 for model input.
Task 4: Data Preprocessing
Goal:
Preprocess the data for training the classifier model.

Description:

Class Balancing: Remove categories with fewer images to avoid bias in the model.
Data Augmentation: Apply transformations like rotation, flipping, and cropping to increase dataset diversity.
Data Splitting: Split the data into training, validation, and test sets using Stratified K-Fold Cross Validation to ensure balanced splits.
Task 5: Build, Train, and Test Classifier Model
Goal:
Build or select a prebuilt classifier model for the task.

Description:

Model: Use MLPClassifier from scikit-learn, a multi-layer perceptron algorithm that trains with backpropagation.
Data Conversion: Convert the data to an appropriate format to feed into the model for training.
Task 6: Analyse and Visualize Results
Goal:
Evaluate the model’s performance and visualize the results.

Description:

Classification Report: Generate a report with precision, recall, and F1 score.
Confusion Matrix: Visualize the model's predictions versus actual labels.
Log Loss: Evaluate the model's uncertainty in its predictions.
