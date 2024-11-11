<!-- ABOUT THE PROJECT -->
## About The Project

This project focuses on classifying leukemia cells into two main types: Acute Lymphoblastic Leukemia (ALL) and Hematopoietic cells (HEM). Acute lymphoblastic leukemia is a prevalent type of childhood cancer, accounting for approximately 25% of all pediatric cancers. The classification task is essential for accurate diagnosis and treatment planning, as distinguishing leukemic blasts from normal cells under a microscope can be challenging due to the morphological similarity of these cells.

The dataset used for this project was sourced from Kaggle and is titled [Leukemia Classification](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification). It consists of segmented cell images taken from microscopic images that closely resemble real-world conditions. These images include variations from staining noise and illumination differences, although most of these have been addressed during data acquisition. Ground truth labels, identifying cells as either ALL or HEM, were annotated by an expert oncologist.

### Project Goals
The goal of this project is to build a robust classification model that can accurately distinguish between ALL and HEM cell types. We implemented and evaluated five different Convolutional Neural Network (CNN) architectures to identify the model with the highest accuracy for this classification task. The architectures tested include:

1. VGG16
2. EfficientNet
3. ResNet
4. DenseNet
5. AlexNet

### Methodology

Data Preparation: Each image in the dataset was labeled and organized to ensure proper dataset structure for training, validation, and testing. The dataset was then split into training, validation, and testing sets using stratified sampling to maintain label proportions across the sets.

Model Design and Training: Each CNN architecture was adapted with a transfer learning approach using pretrained weights, with modifications to optimize for this specific classification task. Models were designed to take 224x224 pixel images as input, and the final layer was adjusted to a binary classification with a softmax activation function to predict ALL or HEM.

Evaluation Metrics: Each model was trained on the training set and evaluated on the validation set. The primary metrics used were:

1. Accuracy: Overall performance metric for classification accuracy.
2. Precision, Recall, and F1-score: To assess performance per class, particularly given the class imbalance.
3. Confusion Matrix: To visualize the distribution of predictions across classes.

### Results
After training each architecture, we obtained the following results on the test set for each model:

- VGG16: Achieved a test accuracy of 92.7%, with precision of 91.5% and recall of 92.3% for both classes.
- EfficientNet: Consistently strong performance with a test accuracy of 94.1%, precision of 93.8%, and recall of 94.0%.
- ResNet: Best performing model with the highest test accuracy of 95.3%, precision of 94.9%, and recall of 95.2%.
- DenseNet: Showed strong performance with a test accuracy of 93.4%, precision of 92.8%, but a slightly lower recall for HEM cells at 91.9%.
- AlexNet: Underperformed relative to the other architectures, achieving a test accuracy of 88.6%, with precision of 86.7% and recall of 87.3%. This model had significant issues distinguishing HEM cells.

### Built With

* [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
* [![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
* [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
* [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
* [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
