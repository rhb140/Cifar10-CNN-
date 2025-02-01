# CIFAR-10 CNN Model Using TensorFlow and Keras From Scratch (Python)

## Description

This project builds a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset using TensorFlow and Keras. The model uses early stopping and learning rate reduction and is trained with data augmentation.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 categories: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. It contains 50,000 training images and 10,000 test images.

## Libraries Used

The following libraries and functions were used in this project:

- **TensorFlow/Keras**:
  - tensorflow.keras.datasets.cifar10.load_data(): Load the CIFAR-10 dataset.
  - tensorflow.keras.models.Sequential(): Create a sequential model.
  - tensorflow.keras.utils.to_categorical(): Convert labels into one-hot encoding.
  - tensorflow.keras.callbacks.EarlyStopping(): Stop training if validation loss does not improve.
  - tensorflow.keras.callbacks.ReduceLROnPlateau(): Reduce learning rate when validation loss stagnates.
  - tensorflow.keras.layers.Conv2D(), MaxPooling2D(), Flatten(), Dense(), Dropout(), BatchNormalization(), Input(): Layers for CNN architecture.
  - tensorflow.keras.preprocessing.image.ImageDataGenerator(): Augment images to enhance generalization.
- **Matplotlib**:
  - matplotlib.pyplot: Plot training and validation accuracy/loss graphs.

## Model Architecture

The CNN architecture consists of:
- **Input Layer**: Accepts 32x32x3 images.
- **Three Convolutional Blocks**, each containing:
  - Two **Conv2D** layers with ReLU activation.
  - **BatchNormalization** to stabilize learning.
  - **MaxPooling2D** to reduce spatial dimensions.
  - **Dropout** to prevent overfitting.
- **Fully Connected Layers**:
  - **Flatten** to convert features into a 1D array.
  - **Dense (512 neurons, ReLU activation, BatchNormalization, Dropout)**.
  - **Output Layer**: 10 neurons with softmax activation (one per class).

## Code Walkthrough

### Data Loading and Preprocessing
```python
# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the data to the range [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```
#### Key Points:
- Load training and test datasets.
- Normalize pixel values from **0-255** to **0-1** for better training stability.
- Convert categorical labels (0-9) to one-hot encoded vectors.

### Data Augmentation
```python
# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
```
Randomly generates several versions of existing data, creating a wider range of features and a larger dataset for training.
#### Key Points:

- **Rotation**: rotates an image up to 15 Degrees
- **Width/Height Shift**: shifts the image up to 10% vertically and horizontally
- **Horizontal Flip**: Flips the image Horizontally

### Create the Model
```python
model = Sequential([
    Input(shape=(32, 32, 3)),
    
    # Conv Block 1
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # Conv Block 2
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Conv Block 3
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Fully Connected Layers
    Flatten(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation="softmax")
])
```
#### Key Points:
- **Conv2D layers** extract features from images.
- **BatchNormalization** adjusts (scales) and stabilizes (normalizes) the activations to help the neural network learn better and faster.
- **MaxPooling2D** reduces dimensions while retaining important information.
- **Dropout** prevents overfitting at different layers.
- **Dense layer** makes the final predictions

### Compile and Train
```python
# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
earlyStop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lrReduction = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Train model
history = model.fit(datagen.flow(X_train, y_train, batch_size=128),
                    epochs=50,
                    validation_data=(X_test, y_test),
                    callbacks=[earlyStop, lrReduction])
```
#### Key Points:
- **Adam optimizer** adapts learning rates dynamically.
- **EarlyStopping** stops training when validation loss stops improving.
- **ReduceLROnPlateau** lowers learning rate when performance reduces.
- **datagen.flow** feeds augmented images to the model.

### Evaluation and Graphs
```python
# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}\nLoss: {loss:.4f}")

# Save the trained model
model.save("cifar10_cnn_model.h5")
print("Model saved successfully.")
```
#### Key Points:
- Evaluate model performance using unseen test data.
- Save trained model for future use.

### Accuracy and Loss Graphs
```python
# Plot Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```
#### Key Points:
Use plt to plot an accuracy and a loss graph.

### Accuracy Graph
![Accuracy Graph](https://github.com/rhb140/Cifar10-CNN-/blob/main/cirfar10Accuracy.jpg?raw=true)

This graph shows both the training and validation accuracy over the epochs.

### Loss Graph
![Loss Graph](https://github.com/rhb140/Cifar10-CNN-/blob/main/cirfar10Loss.jpg?raw=true)

This graph shows both the training and validation loss over the epochs.

## Conclusion

This project demonstrates a deep learning model for classifying CIFAR-10 images. The use of **CNN layers, batch normalization, dropout, data augmentation, early stopping, and learning rate reduction** helps achieve high accuracy reaching 88%.

### Author  
Created by [rhb140](https://github.com/rhb140)

### Citation

CIFAR-10 dataset:Krizhevsky, A., & Hinton, G. (2009). Learning Multiple Layers of Features from Tiny Images. University of Toronto.

