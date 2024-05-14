import fnmatch
import os
from Tensorflow.keras.preprocessing.image import ImageDataGenerator
from Tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_datasets.core.utils.lazy_imports import tensorflow as tf
import tensorflow_datasets.public_api as tfds
_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
_DESCRIPTION = """The Malaria dataset contains a total of 27,558 cell images
with equal instances of parasitized and uninfected cells from the thin blood
smear slide images of segmented cells."""
_NAMES = ["parasitized", "uninfected"]
class Malaria(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(None, None, 3)),
                "label": tfds.features.ClassLabel(names=_NAMES)
            }),
            supervised_keys=("image", "label"),
            homepage="https://lhncbc.nlm.nih.gov/publication/pub9932",
            citation="https://doi.org/10.7717/peerj.456",
        )
    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(_URL)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(extracted_path, "train"),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(extracted_path, "validation"),
                },
            ),
        ]
    def _generate_examples(self, path):
        for name in _NAMES:
            label = 0 if name == "parasitized" else 1
            for root, _, filenames in os.walk(os.path.join(path, name)):
                for filename in fnmatch.filter(filenames, "*.png"):
                    image_path = os.path.join(root, filename)
                    record = {
                        "image": image_path,
                        "label": label
                    }
                    yield image_path,record
data_dir = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
train_dir = os.path.join(data_dir, "train")
validation_dir = os.path.join(data_dir, "validation")
train_datagen= ImageDataGenerator(
    rescale =1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode="categorial"
)
validation_dataset= validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode="binary"
)
model= Sequential([
    Conv2D(32, (3, 3),activation="relu", input_shape=(150,150,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation="relu"),
MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
early_stopping = EarlyStopping(monitor="val_loss", patience=3)
model.fit(
    train_datagen,
    epochs=10,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)
model.save("malaria.h5")
