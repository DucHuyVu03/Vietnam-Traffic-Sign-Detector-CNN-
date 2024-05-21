import cv2 as cv
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def load_data_and_labels(image_folder):
    image_paths = list(Path(image_folder).glob("*.jpg"))
    data = []
    labels = []
    
    for image_path in image_paths:
        # Load the image
        image = cv.imread(str(image_path))
        # Normalize the image
        image = image.astype('float32') / 255.0
        
        # Extract the label from the filename
        label = image_path.stem.split('_')[1]  # Assuming the format is "0123_Label"
        
        data.append(image)
        labels.append(label)
    
    return np.array(data), labels

def encode_labels(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder

npz_output_filename = 'traffic_signs_dataset_128x128'
# Path to the folder containing images
image_folder = 'VietnamTrafficSigns/data_128x128'

# Load data and labels
data, labels = load_data_and_labels(image_folder)

# Encode labels to integers
encoded_labels, encoder = encode_labels(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

# Save the data to a .npz file
np.savez_compressed(f'{npz_output_filename}.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Print some information to verify
print(f"Data shape: {data.shape}")
print(f"Labels shape: {encoded_labels.shape}")
print(f"Unique labels: {len(encoder.classes_)}")
print("Sample labels:", labels[100:120])

# Load the dataset from .npz file
dataset = np.load(f'{npz_output_filename}.npz')
X_train = dataset['X_train']
y_train = dataset['y_train']
X_test = dataset['X_test']
y_test = dataset['y_test']

# Verify the loaded data
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Testing labels shape: {y_test.shape}")