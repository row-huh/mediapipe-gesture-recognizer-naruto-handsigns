
# parallel attempt at extending mediapipe's model maker and teach it to
# classify gestures


# Visualizing Data

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import utils


###########################
# Visualizing the dataset 
###########################

data_root = pathlib.Path("split_dataset/")
dataset_train = data_root / "split_dataset_train"
trainfiles = utils.find_images(dataset_train)

sample_files = np.random.choice(np.asarray(trainfiles), 10)
fig, axarr = utils.plot_image_files(sample_files, ncols=5)

plt.figure(fig)
plt.show()


##########################
# ingesting data and creating training and validation splits
########################

from mediapipe_model_maker.python.vision import gesture_recognizer

handparams = gesture_recognizer.HandDataPreprocessingParams(
    min_detection_confidence=0.5
)
data = gesture_recognizer.Dataset.from_folder(str(dataset_train), handparams)
train_data, validation_data = data.split(0.8)

dataset_test = data_root / "split_data_test"
test_data = gesture_recognizer.Dataset.from_folder(
    str(dataset_test), handparams
)

