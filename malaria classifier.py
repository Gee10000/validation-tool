from tensorflow.keras.datasets import malaria  # Use built-in malaria dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define data augmentation parameters
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    shear_range=0.2,  # Randomly shear images
    zoom_range=0.2,  # Randomly zoom images
    horizontal_flip=True  # Randomly flip images horizontally
)

# Load the malaria dataset
(train_images, train_labels), (test_images, test_labels) = malaria.load_data()

# Reshape images for CNN (assuming grayscale)
train_images = train_images.reshape(-1, 50, 50, 1)  # Reshape to (samples, height, width, channels)
test_images = test_images.reshape(-1, 50, 50, 1)

# Apply data augmentation only to training data
train_datagen.fit(train_images)

# Split training data into training and validation sets (optional)
# You can uncomment and adjust the split ratio as needed
# X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Define CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(50, 50, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")  # Output layer for binary classification
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

# Train the model (uncomment if using validation split)
# model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Train the model on all training data
model.fit(train_images, train_labels, epochs=10, augmentation=datagen, callbacks=[early_stopping])

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

# Save the trained model (optional)
model.save("malaria_classifier.h5")

# You can now use the saved model for prediction on new blood smear images
