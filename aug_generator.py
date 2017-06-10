

import numpy as np
from keras.preprocessing import image
from keras import applications
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from glob import glob
from sklearn.datasets import load_files
from keras.utils import np_utils
from tqdm import tqdm

def preprocessing(img):
    print(img.shape)
    return img

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def save_train_bottleneck_features():
  datagen = image.ImageDataGenerator(rescale=1./255)

  # model = applications.ResNet50(include_top=False, weights='imagenet')
  model = applications.ResNet50(include_top=False)

  x_files, y_data = load_dataset('dogImages/train')
  x_data = paths_to_tensor(x_files)

  x_predict = None
  y_predict = None
  for x_batch, y_batch in datagen.flow(x_data, y_data, batch_size = 32):
      res = model.predict(x_batch)
      if x_predict is None:
          x_predict = res
      else:
        x_predict = np.concatenate((x_predict, res), axis=0)
      if y_predict is None:
          y_predict = y_batch
      else:
          y_predict = np.concatenate((y_predict, y_batch), axis=0)
      print(x_predict.shape[0])
      if x_predict.shape[0] >= 7000:
          break
  print(x_predict.shape)
  print(y_predict.shape)
  np.save('aug_train.npy', x_predict)
  np.save('aug_train_labels.npy', y_predict)

def save_valid_bottleneck_features():
  datagen = image.ImageDataGenerator(rescale=1./255)

  model = applications.ResNet50(include_top=False)

  x_files, y_data = load_dataset('dogImages/valid')
  x_data = paths_to_tensor(x_files)

  x_predict = None
  y_predict = None
  for x_batch, y_batch in datagen.flow(x_data, y_data, batch_size = 5):
      res = model.predict(x_batch)
      if x_predict is None:
          x_predict = res
      else:
        x_predict = np.concatenate((x_predict, res), axis=0)
      if y_predict is None:
          y_predict = y_batch
      else:
          y_predict = np.concatenate((y_predict, y_batch), axis=0)
      print(x_predict.shape[0])
      if x_predict.shape[0] >= 1000:
          break
  print(x_predict.shape)
  print(y_predict.shape)
  np.save('aug_valid.npy', x_predict)
  np.save('aug_valid_labels.npy', y_predict)

  # generator = datagen.flow_from_directory('dogImages/valid',
    # target_size = (224, 224),
    # batch_size = 5,
    # class_mode = 'categorical',
    # shuffle = False)

  # print(generator.class_indices)
  # print(generator.classes)
  # classes_arr = np.array(generator.classes)
  # np.save('labels_validation.npy', classes_arr)
  # bottleneck_features_validation = model.predict_generator(
    # generator,
    # 835 // 5,
    # verbose=1)
  # print(bottleneck_features_validation.shape)
  # np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

def save_train_bottleneck_features_old(d):
  # datagen = image.ImageDataGenerator(rescale=1./255, preprocessing_function=preprocessing)
  datagen = image.ImageDataGenerator(rescale=1./255)

  model = applications.ResNet50(include_top=False, weights='imagenet')

  # num = len(glob(d + '*/*'))
  # print('Number of files in {} is {}'.format(d, num))
  generator = datagen.flow_from_directory(d,
    target_size = (224, 224),
    batch_size = 8,
    class_mode = None,
    shuffle = False)
  # print(generator.classes)
  classes_arr = np.array(generator.classes)
  np.save('labels_train.npy', classes_arr)
  bottleneck_features_train = model.predict_generator(
    generator,
    6680 // 8,
    verbose=1)
  # print(bottleneck_features_train)
  # print(len(bottleneck_features_train))
  print(bottleneck_features_train.shape)
  np.save('bottleneck_features_train.npy', bottleneck_features_train)

save_train_bottleneck_features()
# save_valid_bottleneck_features()

