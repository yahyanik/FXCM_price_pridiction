import tensorflow as tf
import numpy as np
import os
import random
import pathlib
import csv
import pickle
DEBUG = False
import time
import pandas as pd
import matplotlib.pyplot as plt
from window_sourse import timeseries_dataset_from_array

class input_files_generator(object):
    inputs = 'inputs'
    target = "target"

    def __init__(self, path_to_file_list, target_exp, input_exp, resize):
        self.file_list_file = 'fileList.txt'
        # self.file_path_dict = self.generate_files_dict(path_to_file_list)
        self.files_list = self.generate_file_list(path_to_file_list, target_exp, input_exp)
        self.resize = resize


    def read_resize_input_image(self, input_paths):
        images = []
        for path in input_paths:
            images.append(self.read_resize_target_image(path))
            # concatinate along image channels WxHxC
        concat_image = tf.concat(images, -1)
        return concat_image

    def read_resize_target_image(self, target_path):
        img1_file = tf.io.read_file(target_path)
        if DEBUG: print(img1_file.shape)
        img1 = tf.io.decode_image(img1_file, channels=3)
        if DEBUG: print(img1.shape)
        img1.set_shape([None, None, 3])
        if DEBUG: print(img1.shape)
        img1 = tf.image.convert_image_dtype(img1, tf.float32)
        if DEBUG: print(img1.shape)
        img1 = tf.image.resize(img1, size=self.resize)
        if DEBUG: print(img1.shape)
        # img1 = tf.reshape(img1, [1024, 1024, 3])
        # if DEBUG: print(img1.shape)
        return img1

    def generate_file_list(self, path_manifest_file, target, inputs):
        files_list = []
        with open(pathlib.Path(path_manifest_file), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            current_list = []
            directory_path = pathlib.Path(path_manifest_file).parents[0]
            for row in reader:
                current_list.append(pathlib.Path(directory_path, row[target]).as_posix())
                for input in inputs:
                    current_list.append(pathlib.Path(directory_path, row[input]).as_posix())

                files_list.append(current_list)
                current_list = []
        return files_list

    def get_next_data(self):
        while True:
            for element in self.files_list:
                target_image = self.read_resize_target_image(element[0])
                input_image = self.read_resize_input_image(element[1:])
                yield target_image, input_image


def read_image_and_resize(image_path):
    img1_file = tf.io.read_file(image_path)
    if DEBUG: print(img1_file.shape)
    img1 = tf.io.decode_image(img1_file, channels=3)
    if DEBUG: print(img1.shape)
    img1.set_shape([None, None, 3])
    if DEBUG: print(img1.shape)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    if DEBUG: print(img1.shape)
    img_lowres = tf.image.resize_images(img1, size=[256, 256])
    img1 = tf.image.resize_images(img1, size=[640, 640])
    if DEBUG: print(img1.shape)
    img1 = tf.reshape(img1, [640, 640, 3])
    img_lowres = tf.reshape(img_lowres, [256, 256, 3])
    if DEBUG: print(img1.shape)

    return img1, img_lowres


class Multi_Image_Dataset(object):

    def __init__(self, manifest_file_path, target_exp, input_exp, resize, batch_size=1):
        self.resize = resize
        self.batch_size = batch_size
        self.next_element = self.build_iterator(
            input_files_generator(manifest_file_path, target_exp, input_exp, resize))

    def build_iterator(self, input_files_gen: input_files_generator):
        prefetch = 100
        dataset = tf.data.Dataset.from_generator(input_files_gen.get_next_data, output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([self.resize[0], self.resize[1], 3]),
                                                                tf.TensorShape([self.resize[0], self.resize[1], 6])))

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(prefetch)
        self.dataset = dataset

    def _read_images_resize(self, *argv):

        if DEBUG: print(argv)

        imgs = []
        imgs_lowres = []
        for inp in argv[:-1]:
            img, img_lowres = read_image_and_resize(inp)
            imgs.append(img)
            imgs_lowres.append(img_lowres)
        # stack the images along the channels axis
        imgs = tf.stack(imgs, axis=2)
        imgs_lowres = tf.stack(imgs_lowres, axis=2)
        target, target_lowres = read_image_and_resize(argv[-1])
        return ({'inputs': imgs,
                 'inputs_lowres': imgs_lowres,
                 'target': target,
                 'target_lowres': target_lowres})


class DataFromDirectory:
    def __init__(self,
                 path='~/features',
                 batch_size=16, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_DEPTH = 3, repeate=None):
        self.path = path
        self.list_ds = None
        self.ds = None
        self.repeat = repeate
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.batchsize = batch_size
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_DEPTH = IMG_DEPTH
        labeled_ds = self.load_MEF_DATASET()
        self.unbatched_dataset = labeled_ds
        self.ds = self.prepare_for_training(labeled_ds)

    # get the dataset generator
    def get_dataset_generator_obj(self):
        return self.ds

    def get_unbatched_dataset(self):
        return self.unbatched_dataset

    # load model and prepare for training
    def load_MEF_DATASET(self):
        path = self.path
        self.list_ds = tf.data.Dataset.list_files(path)
        labeled_ds = self.list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
        return labeled_ds

    # return the images in in string format
    def process_path(self, file_path):

        # self.file_path = file_path
        # self.non_path = tf.strings.regex_replace(file_path, "Fuzed", "Fuzed_non")
        # img_cud = self.decode_img(self.file_path)
        # img_non_cud = self.decode_img(self.non_path)
        # return img_cud, img_non_cud

        data = np.load(file_path) #TODO make sure that this is the correct way of reading the numpy array
        img = data[0]
        label = data[1]
        img = tf.reshape(img, [self.IMG_DEPTH, self.IMG_HEIGHT, self.IMG_WIDTH])
        return img, label


    # decode image and preprocess
    def decode_img(self, file_path):
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [-1,1] range.
        img = (tf.image.convert_image_dtype(img, tf.float32))
        # resize the image to the desired size.
        img = tf.image.resize(img, [self.IMG_HEIGHT, self.IMG_WIDTH])
        return img

    # make images ready for trainng
    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=50):

        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        # ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        if self.repeat is None:
            # Repeat forever
            ds = ds.repeat()
        else:
            ds = ds.repeat(self.repeat)

        ds = ds.batch(self.batchsize)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

    # get the path tensor
    def get_listpath(self):
        return self.list_ds

    # for checking the speed of the data read
    def timeit(self, steps=1000):
        start = time.time()
        it = iter(self.ds)
        for i in range(steps):
            print(it)
            batch = next(it)
            if i % 10 == 0:
                print('.', end='')
        print()
        end = time.time()

        duration = end - start
        print("{} batches: {} s".format(steps, duration))
        print("{:0.5f} Images/s".format(self.batchsize * steps / duration))


class data_from_generator:
    def __init__(self, path='D:\\Programming\\CUD_global\\Yahya_ImageProcessingCUD\\dataset\\CSOT_DATASET\\trainingSet\\MEF\\',
                 batch_size=16, IMG_HEIGHT=256, IMG_WIDTH=256, repeate=None, WINDOW_SIZE = 2880, LOOK_AHEAD = 2 ,INSTRUMENT_OF_INTEREST = 'EURUSD',hight = 21, width = 8, target = None):

        self.path = path
        self.f = None
        self.list_ds = None
        self.ds = None
        # result = self.load_obj(self.path + 'res.pkl')
        self.WINDOW_SIZE = WINDOW_SIZE
        self.LOOK_AHEAD = LOOK_AHEAD
        self.target = target
        self.INSTRUMENT_OF_INTEREST = INSTRUMENT_OF_INTEREST
        self.repeat = repeate
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.batchsize = batch_size
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.hight = hight
        self.width = width
        self.combined = self.windowed(self.path + 'res.pkl')



        # self.labeled_ds = self.dataset(self.combined, self.INSTRUMENT_OF_INTEREST)
        # self.ds = self.prepare_for_training(self.labeled_ds)

    # def get_generator(self):
    #     return

    def get_dataset_generator_obj(self):
        return self.ds

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=50):

        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        if self.repeat is None:
            # Repeat forever
            ds = ds.repeat()
        else:
            ds = ds.repeat(self.repeat)

        ds = ds.batch(self.batchsize)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

    def dataset(self, result, instrument):
        ds_counter = tf.data.Dataset.from_generator(self.get_labels, args=[result, instrument], output_types=(tf.float16, tf.float16), output_shapes=([None, None, None], (3,)))
        return ds_counter

    def get_labels(self, combined_features, instrument, train, test, batch_size, training = True):

        if instrument == 'EURUSD':
            if training:
                iii = 0
                teta = train
            else:
                iii = 0+train
                teta = train+test
            while iii < teta:
                # window_features = features[i:i+WINDOW_SIZE]
                l_combined_window_features = []
                l_labels = []
                for k in range(batch_size):
                    combined_window_features = combined_features[iii:(iii + self.WINDOW_SIZE)]
                    l_combined_window_features.append(combined_window_features.to_numpy(dtype=np.float16))


                    temp1 = combined_features['BidOpen'+instrument][iii+self.WINDOW_SIZE+self.LOOK_AHEAD]
                    temp2 = combined_features['AskClose'+instrument][iii+self.WINDOW_SIZE+self.LOOK_AHEAD]

                    if combined_window_features['AskClose'+instrument][-1] < temp1:
                        label = [1, 0, 0]  # buy
                    elif combined_window_features['BidOpen'+instrument][-1] > temp2:
                        label = [0, 0, 1]  # sell
                    else:
                        label = [0, 1, 0]  # do not do anything
                    l_labels.append(np.array([label], dtype=np.float16))
                    iii+=1

                f = np.concatenate(l_combined_window_features, axis=0)
                f = f.reshape([batch_size, self.WINDOW_SIZE, self.width, self.hight, 1])
                yield f, np.concatenate(l_labels, axis=0)

    def windowed (self, path):
        if self.target is None:
            l=[]
            dict = self.load_obj(path)
            for k in dict:
                df = dict[k]
                print('lengh:', k, len(df))
                df.set_index('DateTime', inplace=True)
                # df = df.drop(columns=['DateTime'])
                print([colum for colum in df.columns])
                l.append(df[['BidOpen'+k, 'BidClose'+k, 'AskOpen'+k, 'AskClose'+k]])
            combined_features = pd.concat(l, axis=1, join='inner')
            print(len(combined_features.columns), len(combined_features))
            temp = combined_features.to_numpy(dtype=np.float16)
            f = temp.reshape([combined_features.shape[0], self.width, self.hight, 1])
            np.save('./_features.npy', f)
        else:
            print('already have features ready')
            combined_features = np.load(self.path +'_features.npy')
        return combined_features

    def save_obj(self,obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self,name):
        with open(name , 'rb') as f:
            return pickle.load(f)

    def rolling_window(self, a, window):
        shape = (a.shape[0]- window + 1, window) + a.shape[1:]
        strides = (a.strides[0],) + a.strides
        print(strides)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    # def npy_to_tfrecords(self, feature, label, name):
    #     # write records to a tfrecords file
    #     writer = tf.python_io.TFRecordWriter('~/'+name)
    #
    #     # Loop through all the features you want to write
    #     for i in range(feature):
    #         features = {}
    #         features['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=feature[i,:,:,:,].flatten()))
    #         features['y'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label[i,:]))
    #         example = tf.train.Example(features=tf.train.Features(feature=features))
    #
    #         # Serialize the example to a string
    #         serialized = example.SerializeToString()
    #
    #         # write the serialized objec to the disk
    #         writer.write(serialized)
    #     writer.close()

    def keras_based_window(self, instrument, batch_size, start=0, validation_per=0.08, test_per=0.08, SHUFFLE_BUFFER_SIZE=500):

        if (self.target is None) or (self.target.shape[0] != (len(self.combined) - self.WINDOW_SIZE - self.LOOK_AHEAD)):
            temp = self.combined.to_numpy(dtype=np.float16)
            self.f = temp.reshape([self.combined.shape[0], self.width, self.hight])
            l = []
            if instrument == 'EURUSD':
                for iii in range(0, len(self.combined) - self.WINDOW_SIZE - self.LOOK_AHEAD):
                    combined_window_features = self.combined[iii:(iii + self.WINDOW_SIZE)]
                    temp1 = self.combined['BidOpen' + instrument][iii + self.WINDOW_SIZE + self.LOOK_AHEAD]
                    temp2 = self.combined['AskClose' + instrument][iii + self.WINDOW_SIZE + self.LOOK_AHEAD]

                    if combined_window_features['AskClose' + instrument][-1] < temp1:
                        label = [1, 0, 0]  # buy
                    elif combined_window_features['BidOpen' + instrument][-1] > temp2:
                        label = [0, 0, 1]  # sell
                    else:
                        label = [0, 1, 0]  # do not do anything

                    l.append(np.array([label], dtype=np.float16))
            self.target = np.concatenate(l, axis=0)
            np.save('./_labels.npy', self.target)

        try:
            kol = len(self.combined)
            print(self.f[:(len(self.combined) - self.WINDOW_SIZE - self.LOOK_AHEAD), :, :].shape)
        except:
            kol = self.combined.shape[0]
            print(self.combined[:((self.combined.shape[0]) - self.WINDOW_SIZE - self.LOOK_AHEAD), :, :].shape)

        offset = self.WINDOW_SIZE + self.LOOK_AHEAD - 1
        start_train = start
        end_train = int(kol - test_per*kol - validation_per*kol)
        start_valid = int(kol - test_per*kol - validation_per*kol)
        end_valid = int(kol - test_per*kol)
        start_test = int(kol - test_per*kol)
        end_test = kol - offset

        print('This is the shape of the target', self.target.shape)

        # try:
        #     np.load('/home/yahya/_label_test_slice.npy')
        # except:
        #     feature_train_slice = self.rolling_window(self.combined[(start_train):(end_train+self.WINDOW_SIZE-1), :, :, :], self.WINDOW_SIZE)
        #     label_train_slice = self.target[(start_train):(end_train),:]
        #
        #     feature_test_slice = self.rolling_window(self.combined[(start_test):(end_test+self.WINDOW_SIZE-1 -1), :, :, :], self.WINDOW_SIZE)
        #     label_test_slice = self.target[(start_test):(end_test),:]
        #     np.save('/home/yahya/_feature_train_slice.npy', feature_train_slice)
        #     np.save('/home/yahya/_label_train_slice.npy', label_train_slice)
        #     np.save('/home/yahya/_feature_test_slice.npy', feature_test_slice)
        #     np.save('/home/yahya/_label_test_slice.npy', label_test_slice)



            # feature_kol_dataset = tf.data.Dataset.from_tensor_slices(self.combined[(start_train):(end_train+self.WINDOW_SIZE-1), :, :, :])
            # print(feature_kol_dataset.element_spec)
            # feature_kol_dataset = feature_kol_dataset.window(self.WINDOW_SIZE, shift=1, stride=1, drop_remainder=False)
            # print(feature_kol_dataset.element_spec)
            # label_kol_dataset = tf.data.Dataset.from_tensor_slices(self.target[(start_train):(end_train),:])
            # train_kol_dataset = tf.data.Dataset.zip((feature_kol_dataset, label_kol_dataset)).shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
            # print(train_kol_dataset.element_spec)



        # feature_train_slice = np.load('/home/yahya/_feature_train_slice.npy', mmap_mode='r+', allow_pickle=False, fix_imports=True, encoding='ASCII')
        # label_train_slice = np.load('/home/yahya/_label_train_slice.npy', mmap_mode='r+', allow_pickle=False, fix_imports=True, encoding='ASCII')
        # feature_test_slice = np.load('/home/yahya/_feature_test_slice.npy', mmap_mode='r+', allow_pickle=False, fix_imports=True, encoding='ASCII')
        # label_test_slice = np.load('/home/yahya/_label_test_slice.npy', mmap_mode='r+', allow_pickle=False, fix_imports=True, encoding='ASCII')
        #
        # # train_kol_dataset = train_kol_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        # #
        # # train_kol_dataset = tf.data.Dataset.from_generator(gen_train, tf.float16)
        # # train_kol_dataset =train_kol_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        # # test_dataset = tf.data.Dataset.from_generator(gen_test, tf.float16)
        # # test_dataset = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        # train_kol_dataset = tf.data.Dataset.from_tensor_slices((feature_train_slice, label_train_slice))
        # train_kol_dataset = train_kol_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        # test_dataset = tf.data.Dataset.from_tensor_slices((feature_test_slice, label_test_slice))
        # test_dataset = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        #
        # train_dataset = train_kol_dataset.take((int(kol - test_per*kol) - int(validation_per*(kol - test_per*kol))))
        # valid_dataset = train_kol_dataset.skip(int(validation_per*(kol - test_per*kol)))
        #
        # test_dataset = test_dataset.batch(batch_size)
        # train_dataset = train_dataset.batch(batch_size)
        # valid_dataset = valid_dataset.batch(batch_size)

        test_dataset_size = end_test - start_test -1
        train_dataset_size = (int(kol - test_per*kol) - int(validation_per*(kol - test_per*kol)))
        valid_dataset_size = int(validation_per*(kol - test_per*kol))

        try:
            print(self.f[:(len(self.combined) - self.WINDOW_SIZE - self.LOOK_AHEAD), :, :].shape)
            train_kol_dataset = timeseries_dataset_from_array(
                    self.f[:(len(self.combined) - self.WINDOW_SIZE - self.LOOK_AHEAD), :, :, :],
                    self.target, self.WINDOW_SIZE, sequence_stride=1, sampling_rate=1,
                    batch_size=batch_size, shuffle=True, seed=None, start_index=start_train,
                    end_index=end_train)


            test_dataset = timeseries_dataset_from_array(
                    self.f[:(len(self.combined) - self.WINDOW_SIZE - self.LOOK_AHEAD), :, :, :],
                    self.target, self.WINDOW_SIZE, sequence_stride=1, sampling_rate=1,
                    batch_size=batch_size, shuffle=False, seed=None, start_index=start_test,
                    end_index=end_test)
        except:
            print(self.combined[:((self.combined.shape[0]) - self.WINDOW_SIZE - self.LOOK_AHEAD), :, :, :].shape)

            train_dataset = timeseries_dataset_from_array(
                    self.combined[start_train:end_train, :, :, :],
                    self.target[start_train:end_train], sequence_length=self.WINDOW_SIZE, batch_size = batch_size, shuffle=True,)

            valid_dataset = timeseries_dataset_from_array(
                self.combined[start_valid:end_valid, :, :, :],
                self.target[start_valid:end_valid], sequence_length=self.WINDOW_SIZE, batch_size=batch_size, shuffle=True,)

            test_dataset = timeseries_dataset_from_array(
                    self.combined[start_test:end_test-1, :, :, :],
                    self.target[start_test:end_test], sequence_length=self.WINDOW_SIZE, batch_size=batch_size, shuffle=False,)

        # train_kol_dataset = timeseries_dataset_from_array(
        #         self.combined[(start_train) : (end_train + self.WINDOW_SIZE-1), :, :, :],
        #         self.target[(start_train):(end_train),:], self.WINDOW_SIZE, sequence_stride=1, sampling_rate=1,
        #         batch_size=batch_size, shuffle=False, seed=None)
        #
        # test_dataset = timeseries_dataset_from_array(
        #         self.combined[start_test: (end_test + self.WINDOW_SIZE-1), :, :, :],
        #         self.target[(start_test):(end_test),:], self.WINDOW_SIZE, sequence_stride=1, sampling_rate=1,
        #         batch_size=batch_size, shuffle=False, seed=None)

        # train_kol_dataset = train_kol_dataset.unbatch()
        # train_kol_dataset = train_kol_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        # train_dataset = train_kol_dataset.take((int(kol - test_per * kol) - int(validation_per * (kol - test_per * kol))))
        # valid_dataset = train_kol_dataset.skip(int(validation_per*(kol - test_per*kol)))

        # train_dataset = train_dataset.batch(batch_size)
        # valid_dataset = valid_dataset.batch(batch_size)

        w = (train_dataset, valid_dataset,test_dataset, train_dataset_size, valid_dataset_size, test_dataset_size)
        # w = (train_dataset, valid_dataset, train_dataset_size, valid_dataset_size)
        return w

if __name__ == "__main__":
    dataset_iter = Multi_Image_Dataset(r'C:\Users\amnikhil\Desktop\official_test_dataset\formatted_manifest.csv',
                                       'target0', ['exp0', 'exp3'])
    dataset = dataset_iter.dataset

    for target, input_img in dataset.take(1):
        print(target.shape)
        print(input_img.shape)
        plt.imshow(input_img[0, :, :, 3:])
        plt.imshow(target[0, :, :, :])
        plt.show()
