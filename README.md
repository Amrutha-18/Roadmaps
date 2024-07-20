## Road Segmentation Model

This repository contains a road segmentation model built using a U-Net architecture in TensorFlow/Keras. The model is designed to perform pixel-level segmentation of road images.

## Table of Contents
To use this project, you need to have Python installed along with the required libraries. You can install the required packages using pip:

pip install numpy tensorflow matplotlib

## Dataset Preparation
The model expects images and corresponding masks to be placed in specific directories:

Train Dataset: /Users/amruthapullagummi/Downloads/train

Validation Dataset: /Users/amruthapullagummi/Downloads/valid

Test Dataset: /Users/amruthapullagummi/Downloads/test

## Each directory should contain:
.jpg files for satellite images.

.png files for masks (for training).

Ensure that images and masks have the same filenames.

## Model Architecture

The model is built using a U-Net architecture, which includes:

*Downsampling Path*: Convolutional layers followed by max-pooling. 

*Bottleneck*: Convolutional layers at the lowest resolution.

*Upsampling Path*: Upsampling layers followed by concatenation with corresponding downsampling layers and convolutional layers.

The final output is a segmentation map with the same dimensions as the input image.

## Training

To train the model, adjust the paths in the script to point to your dataset directories and run the following:

Load the datasets.

Initialize and compile the U-Net model.

Train the model with the specified hyperparameters.

Batch Size: 2 (tried with 4 , but the kernel was dying due to excess RAM usage)

Patch Size: 256x256 

Epochs: 5 ( Can try 10 too)

Learning Rate: 0.0001 (tried with 0.001 the model was over fitting)

You can adjust these parameters in the file as needed.


## Training the Model:

Update the train_dir, valid_dir, and test_dir variables in the script with your dataset paths and execute the training script.

## Visualizing Predictions:

Use the provided visualize_predictions function to see the model's predictions.

Use the saved weights file and Load the Model

The model is saved using:

model.save('/Users/amruthapullagummi/Downloads/road_segmentation_model.h5') in the code.

## To load the model later:

from tensorflow.keras.models import load_model

model = load_model('/Users/amruthapullagummi/Downloads/road_segmentation_model.h5')

The trained model can be saved to a file and reloaded later for inference or further training


*Save model*
model.save('/path/to/road_segmentation_model.h5')


*Load model*
from tensorflow.keras.models import load_model

model = load_model('/path/to/road_segmentation_model.h5')
