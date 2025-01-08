# Convolutional Neural Network


# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../../../datasets/cnn_dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('../../../datasets/cnn_dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Building the CNN
cnn = tf.keras.models.Sequential()
# Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Flattening
cnn.add(tf.keras.layers.Flatten())
# Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Training the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = cnn.fit(x = training_set, validation_data = test_set, epochs = 20)


# Plotting Accuracy and Loss
plt.figure(figsize=(12, 5))
# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_metrics.png')  # Save the plot
plt.show()

# Testing Predictions on Random Test Images
def predict_image(image_path, model, class_indices):
    show_img = image.load_img(image_path)
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    prediction = model.predict(img_array)
    predicted_class = 'Cat' if prediction[0][0] < 0.5 else 'Dog'
    true_class = image_path.split('/')[-2]  # Assuming folder structure for class label
    plt.imshow(show_img)
    plt.title(f"True: {true_class}, Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

# Predicting on a few test images
test_images = [
    '../../../datasets/cnn_dataset/test_set/cats/cat.4500.jpg',
    '../../../datasets/cnn_dataset/test_set/cats/cat.4520.jpg',
    '../../../datasets/cnn_dataset/test_set/cats/cat.4234.jpg',
    '../../../datasets/cnn_dataset/test_set/cats/cat.4123.jpg',
    '../../../datasets/cnn_dataset/test_set/cats/cat.4448.jpg',
    '../../../datasets/cnn_dataset/test_set/dogs/dog.4447.jpg',
    '../../../datasets/cnn_dataset/test_set/dogs/dog.4540.jpg',
    '../../../datasets/cnn_dataset/test_set/dogs/dog.4525.jpg',
    '../../../datasets/cnn_dataset/test_set/dogs/dog.4334.jpg',
    '../../../datasets/cnn_dataset/test_set/dogs/dog.4155.jpg',
]
for img_path in test_images:
    predict_image(img_path, cnn, training_set.class_indices)
