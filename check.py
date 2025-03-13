import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Load your pre-trained model
model = tf.keras.models.load_model("G:\\model\\gmod.keras")

# Inspect the model architecture to understand the expected input shape
model.summary()

# Adjust image dimensions based on the model's expected input
img_height = 224  # Changed to 224
img_width = 224   # Changed to 224
color_mode = 'rgb'  # Use 'rgb' since the model expects 3 channels

# Set the batch size
batch_size = 30

# Specify the directory containing your dataset
target_dir = r"C:\Users\eyita\Downloads\archive (1)\BreaKHis_v1\BreaKHis_v1\histology_slides\breast"

# List the contents of the target directory
print(os.listdir(target_dir))

# Manually specify the class names to exclude non-class files
class_names = ['benign', 'malignant']

# Create the training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    target_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode=color_mode,
    class_names=class_names  # Corrected argument name
)

# Create the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    target_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode=color_mode,
    class_names=class_names  # Corrected argument name
)

# Get the class names
print(class_names)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

# Define a function for preprocessing images
def preprocess_image(img_path):
    img = tf.keras.utils.load_img(
        img_path,
        target_size=(img_height, img_width),
        color_mode=color_mode  # Ensure images have 3 channels
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values if required
    img_array = np.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

# Correct the file path and handle any escape sequences
local_image_path = r"C:\Users\eyita\Downloads\idk.png"

# Preprocess the image
img_array = preprocess_image(local_image_path)

# Make predictions
predictions = model.predict(img_array)

# Handle predictions based on the number of classes
if num_classes == 2:
    # Binary classification
    probability = tf.nn.sigmoid(predictions[0])[0]
    confidence = probability * 100
    class_index = 0 if probability < 0.5 else 1
else:
    # Multi-class classification
    probability = tf.nn.softmax(predictions[0])
    confidence = 100 * np.max(probability)
    class_index = np.argmax(probability)

# Output the prediction
print(f"This image most likely belongs to '{class_names[class_index]}' with a {confidence:.2f}% confidence.")
