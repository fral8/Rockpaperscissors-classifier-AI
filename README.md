# AI Rock-Paper-Scissors Image Classifier

## Approach

This repository contains code for an AI Rock-Paper-Scissors Image Classifier using a Convolutional Neural Network (CNN). The model is built using TensorFlow and Keras and aims to classify images of hands showing rock, paper, or scissors gestures. The dataset is divided into training and validation sets, and data augmentation is applied to improve model performance.

## Dataset

The dataset used for training and validation consists of images of hands displaying rock, paper, or scissors gestures. The images are organized into the following directories:

- `training/rock`: Contains training images of hands showing the rock gesture.
- `training/paper`: Contains training images of hands showing the paper gesture.
- `training/scissors`: Contains training images of hands showing the scissors gesture.
- `validation/rock`: Contains validation images of hands showing the rock gesture.
- `validation/paper`: Contains validation images of hands showing the paper gesture.
- `validation/scissors`: Contains validation images of hands showing the scissors gesture.

## Data Augmentation

Data augmentation techniques are applied to the training data to enhance the model's ability to generalize. These techniques include rotation, width and height shifts, shearing, zooming, and horizontal flipping.

## Model Architecture

The AI Rock-Paper-Scissors Image Classifier model consists of a series of Convolutional and MaxPooling layers followed by Dense layers. The final layer uses a softmax activation function to predict one of the three classes: rock, paper, or scissors.

## Training

The model is trained using the training data and validated using the validation data. The training process involves minimizing the categorical cross-entropy loss using the RMSprop optimizer with a learning rate of 0.001. The training will stop early if the training accuracy reaches 92%.

## Training Performance

The training and validation accuracy and loss are plotted to visualize the model's performance during training.

## Usage

To use the AI Rock-Paper-Scissors Image Classifier, you can load the saved model and call its `predict` method on new images to classify them as rock, paper, or scissors.

Feel free to experiment with different hyperparameters, add more layers, or adjust the data augmentation techniques to improve the model's accuracy further.

For any questions or suggestions, please contact [Francesco Alotto](mailto:franalotto94@gmail.com). Have fun playing Rock-Paper-Scissors with AI! 🤖✊✋✌️
