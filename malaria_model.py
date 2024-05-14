import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = "C:\\Users\\administrator\\Downloads\\cell_images\\"
train_dir = os.path.join(data_dir, 'Train') 
validation_dir = os.path.join(data_dir, 'Validation')
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))),
model.add(MaxPooling2D((2, 2))),
model.add(Conv2D(64, (3, 3), activation='relu')),
model.add(MaxPooling2D((2, 2))),
model.add(Conv2D(128, (3, 3), activation='relu')),
model.add(MaxPooling2D((2, 2))),
model.add(Conv2D(128, (3, 3), activation='relu')),
model.add(MaxPooling2D((2, 2))),
model.add(Flatten()),
model.add(Dropout(0.5)),
model.add(Dense(512, activation='relu')),
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)
model.save('malaria_model.h5')