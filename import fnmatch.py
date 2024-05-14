import fnmatch
import os

from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"

_DESCRIPTION = """The Malaria dataset contains a total of 27,558 cell images
with equal instances of parasitized and uninfected cells from the thin blood
smear slide images of segmented cells."""

_NAMES = ["parasitized", "uninfected"]

_IMAGE_SHAPE = (None, None, 3)


class Builder(tfds.core.GeneratorBasedBuilder):
  """Malaria Cell Image Dataset Class."""

  VERSION = tfds.core.Version("1.0.0")

  def _info(self):
    """Define Dataset Info."""

    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(shape=_IMAGE_SHAPE),
            "label": tfds.features.ClassLabel(names=_NAMES),
        }),
        supervised_keys=("image", "label"),
        homepage="https://lhncbc.nlm.nih.gov/publication/pub9932",
    )

  def _split_generators(self, dl_manager):
    """Define Splits."""

    path = dl_manager.download_and_extract(_URL)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "data_dir_path": os.path.join(path, "cell_images"),
            },
        ),
    ]

  def _generate_examples(self, data_dir_path):
    """Generate images and labels for splits."""
    folder_names = ["Parasitized", "Uninfected"]

    for folder in folder_names:
      folder_path = os.path.join(data_dir_path, folder)
      for file_name in tf.io.gfile.listdir(folder_path):
        if fnmatch.fnmatch(file_name, "*.png"):
          image = os.path.join(folder_path, file_name)
          label = folder.lower()
          image_id = "%s_%s" % (folder, file_name)
          yield image_id, {"image": image, "label": label}