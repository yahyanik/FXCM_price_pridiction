

from ConvLSTM import ConvLSTM
##3706429440
from data_handler import *
# from keras.callbacks import EarlyStopping
import numpy
import pathlib
import os

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# print("All devices: ", tf.config.list_logical_devices('TPU'))
# strategy = tf.distribute.TPUStrategy(resolver)

p = os.getcwd() + '/'
# tf.keras.backend.set_floatx('float16')
# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
DATA_ADDRESS = p + 'Windowded'
BATCH_SIZE = 256
EPOCH = 50
HEIGHT = 4
WIDTH = 21
DEPTH = 60 #a week of information 7*24*60 # 2 days: 2880, 10 hours: 600
NAME = 'FXCM_LookUp_LeakyReLU_weighted_0.003'
CHECKPOINT_DIR = p + 'logs/'+NAME+'/'
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0003
LOOK_AHEAD = 1 # how much we want to predict, 2 hours
NUM_SAMPLES = 2757760
INSTRUMENT_OF_INTEREST = 'EURUSD'
LOG_DIR = CHECKPOINT_DIR
CHECK_RATE = 1
IGNORE_DATA = 0
DROP_OUT_SCORE = 0.5
SHUFFLE_BUFFER_SIZE = 5000
MODEL_PATH = None#CHECKPOINT_DIR+NAME+'_7.h5' #os.path.join(os.getcwd(), NAME+'_2.h5')

try:
    TARGETS = np.load( p + '_labels.npy').astype(np.uint32)
    print('target is read!')
except:
    print('target was not found')
    TARGETS = None
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()

#
print('calling dataset generator')
data_obj = data_from_generator (path=p, batch_size=BATCH_SIZE, WINDOW_SIZE = DEPTH, hight = HEIGHT, width = WIDTH, target = TARGETS, LOOK_AHEAD = LOOK_AHEAD)

validation_per = 0.1
test_per = 0.1
print('making generator for train and test')
train_dataset, valid_dataset,test_dataset, train_dataset_size, valid_dataset_size, test_dataset_size = \
    data_obj.keras_based_window(INSTRUMENT_OF_INTEREST, BATCH_SIZE, start=IGNORE_DATA, validation_per=validation_per,
                                test_per=test_per, SHUFFLE_BUFFER_SIZE=SHUFFLE_BUFFER_SIZE)



# train_dataset, valid_dataset, train_dataset_size, valid_dataset_size = \
#     data_obj.keras_based_window(INSTRUMENT_OF_INTEREST, BATCH_SIZE, start=IGNORE_DATA, validation_per=validation_per,
#                                 test_per=test_per, SHUFFLE_BUFFER_SIZE=SHUFFLE_BUFFER_SIZE)

# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()

# valid_dataset, valid_samples = data_obj.keras_based_window(INSTRUMENT_OF_INTEREST, BATCH_SIZE, IGNORE_DATA+train, start_point= IGNORE_DATA, dataset_type='valid', validation_num = Validation)

# for kk in train_dataset.take(1):
#     print(kk[0].shape)

# print('breaking dataset to validation and training')
# Splitting the dataset for training and testing.
# def is_test(x, _):
#     return x % 10 == 0
#
#
# def is_train(x, y):
#     return not is_test(x, y)
#
#
# recover = lambda x, y: y
#
# # Split the dataset for training.
# valid_dataset = training_dataset[0].enumerate() \
#     .filter(is_test) \
#     .map(recover)

# Split the dataset for testing/validation.
# train_dataset = training_dataset[0].enumerate() \
#     .filter(is_train) \
#     .map(recover)

print('dataset_ready')
label_0 = 0
label_1 = 0
label_2 = 0
for _, label in train_dataset.take(int(train_dataset_size / BATCH_SIZE)):
    label_0+=np.count_nonzero(label.numpy()[:,0] == 1)
    label_1 += np.count_nonzero(label.numpy()[:, 1] == 1)
    label_2 += np.count_nonzero(label.numpy()[:, 2] == 1)

total = label_0+label_1+label_2
weight_for_0 =(1 / label_0)*(total)/2.0
weight_for_1 =(1 / label_1)*(total)/2.0
weight_for_2 =(1 / label_2)*(total)/2.0
class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
#
print('calling model')
# # with strategy.scope():
# try:
model_obj = ConvLSTM((DEPTH, WIDTH, HEIGHT, 1), weight_decay=WEIGHT_DECAY, checkpoint_dir=CHECKPOINT_DIR,
                     is_trainng=True, name=NAME, batch_size=BATCH_SIZE, log_dir=LOG_DIR, drop_out_score=DROP_OUT_SCORE, model_path=MODEL_PATH)
# except:
#     model_obj = ConvLSTM((DEPTH, WIDTH, HEIGHT, 1), weight_decay=WEIGHT_DECAY, checkpoint_dir=CHECKPOINT_DIR,
#                          is_trainng=True, name=NAME, batch_size=BATCH_SIZE, log_dir=LOG_DIR,
#                          drop_out_score=DROP_OUT_SCORE)
model_obj.compile_adam(learning_rate=LEARNING_RATE)

model_obj.fit(train_dataset.repeat(), epochs=EPOCH,
              batch_size=BATCH_SIZE, num_sample= train_dataset_size, testing_data=valid_dataset.repeat(),
               validation_steps=valid_dataset_size, history_loc = CHECKPOINT_DIR+'history.p', class_weight=class_weight)


 ########test: #############



























# print('out')
# t = []
# for i, l in training_dataset.take(2):
#     print(i.numpy().shape)
#     t.append(i)
#     print(l.numpy())
# if np.array_equal(t[0], t[1]):
#     print('bad')
# else:
#     print('ok')












































'''
kol = NUM_SAMPLES - WINDOW_SIZE - LOOK_AHEAD
train = int((kol- 0.01*kol)/BATCH_SIZE)
test = int((kol - train)/BATCH_SIZE)



# training_data = data_obj.get_labels(dataset, INSTRUMENT_OF_INTEREST, train, BATCH_SIZE, training=True)
# dataset = tf.data.Dataset.from_generator(data_obj.get_labels,args=[dataset, INSTRUMENT_OF_INTEREST, train, BATCH_SIZE, True], output_types=(tf.float32, tf.float32),output_shapes =(tf.TensorShape([None, None, 21, 8, 1]), tf.TensorShape([None, 3])))
# dataset_test = tf.data.Dataset.from_generator(data_obj.get_labels,args=[dataset, INSTRUMENT_OF_INTEREST, test, BATCH_SIZE, False], output_types=(tf.float32, tf.float32),output_shapes =(tf.TensorShape([None, None, 21, 8, 1]), tf.TensorShape([None, 3])))

print(list(dataset.take(2).numpy()))

dataset = data_obj.get_labels(dataset, INSTRUMENT_OF_INTEREST, train, test, BATCH_SIZE, training=True)
testing_data = data_obj.get_labels(dataset, INSTRUMENT_OF_INTEREST, train, test, BATCH_SIZE, training=False)
#
# model_obj.fit(dataset, epochs=EPOCH, num_sample=NUM_SAMPLES, batch_size=BATCH_SIZE, testing_data=testing_data, validation_steps = test)
print('BEGIN TRAINING')
for e in range(EPOCH):
    mt = 0
    for b in range(train):
        f, l = next(dataset)
        temp = model_obj.train_on_batch(b, f, l)
        mt += temp
        print('batch #', b, 'error: ', temp)

    if (EPOCH%CHECK_RATE) == 0:
        mm = 0
        for bb in range(test):
            f, l = next(testing_data)
            mm += model_obj.eval_training(f, l)
        print('TESTING ERROR:', mm/test)

    print('EPOCH', e, 'IS FINISHED', 'ERROR: ', mt/train)
    print('SAVING MODEL: ...')
    model_obj.save(e)

'''

































# f = []
# for a in range(2):
#     i ,w = next(dataset)
#     print (i.shape)
#     print(w)
#     f.append(i)
# if np.array_equal(f[0], f[1]):
#   print('bad')
#
# else:
#   print('ok')










# for _ in range (1):#EPOCH
  # iii = 0
  # for i, w in data_obj.get_labels(data_obj.combined, INSTRUMENT_OF_INTEREST, kol, BATCH_SIZE):
    # if iii < train:
    #   f.append(i)
    # else:
    #   kk
    # if iii >= kol:
    #   break
    # iii+=BATCH_SIZE


    # print (i.shape)
    # print(w)

# if np.array_equal(f[0], f[1]):
#   print('bad')
#
# else:
#   print('ok')

# dataset = data_obj.get_dataset_generator_obj()
#
# for count_batch in data_obj.labeled_ds.batch(2).take(2):
#   print(count_batch.numpy().shape)
#   print(count_batch.numpy())
#
# for count_batch in dataset.take(2):
#   print(count_batch.numpy().shape)
#   print(count_batch.numpy())








