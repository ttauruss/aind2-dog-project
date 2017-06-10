

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications

def save_valid_bottleneck_features():
  datagen = ImageDataGenerator(rescale=1./255)

  model = applications.ResNet50(include_top=False, weights='imagenet')

  generator = datagen.flow_from_directory('dogImages/valid',
    target_size = (224, 224),
    batch_size = 16,
    class_mode = None,
    shuffle = False)
  bottleneck_features_validation = model.predict_generator(
    generator,
    835 // 16)
  np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

def save_train_bottleneck_features():
  datagen = ImageDataGenerator(rescale=1./255)

  model = applications.ResNet50(include_top=False, weights='imagenet')

  generator = datagen.flow_from_directory('dogImages/train',
    target_size = (224, 224),
    batch_size = 16,
    class_mode = None,
    shuffle = False)
  bottleneck_features_train = model.predict_generator(
    generator,
    6680 // 16)
  np.save('bottleneck_features_train.npy', bottleneck_features_train)

save_train_bottleneck_features()
