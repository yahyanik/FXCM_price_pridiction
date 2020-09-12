from ConvLSTM import ConvLSTM
from data_handler import *
from sklearn.metrics import multilabel_confusion_matrix


p = os.getcwd() + '/'
DATA_ADDRESS = p + 'Windowded'
BATCH_SIZE = 256
EPOCH = 50
HEIGHT = 4
WIDTH = 21
DEPTH = 60 #a week of information 7*24*60 # 2 days: 2880, 10 hours: 600
NAME = 'FXCM_LookUp_LeakyReLU'
CHECKPOINT_DIR = p + 'logs/'+NAME+'/'
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
LOOK_AHEAD = 1 # how much we want to predict, 2 hours
NUM_SAMPLES = 2757760
INSTRUMENT_OF_INTEREST = 'EURUSD'
LOG_DIR = CHECKPOINT_DIR
CHECK_RATE = 1
IGNORE_DATA = 0
DROP_OUT_SCORE = 0.5
SHUFFLE_BUFFER_SIZE = 5000
MODEL_PATH = CHECKPOINT_DIR+NAME+'_8.h5' #os.path.join(os.getcwd(), NAME+'_2.h5')

try:
    TARGETS = np.load( p + '_labels.npy').astype(np.uint32)
    print('target is read!')
except:
    print('target was not found')
    TARGETS = None

print('calling dataset generator')
data_obj = data_from_generator (path=p, batch_size=BATCH_SIZE, WINDOW_SIZE = DEPTH, hight = HEIGHT, width = WIDTH, target = TARGETS, LOOK_AHEAD = LOOK_AHEAD)

validation_per = 0.1
test_per = 0.1
print('making generator for train and test')
train_dataset, valid_dataset,test_dataset, train_dataset_size, valid_dataset_size, test_dataset_size = \
    data_obj.keras_based_window(INSTRUMENT_OF_INTEREST, BATCH_SIZE, start=IGNORE_DATA, validation_per=validation_per,
                                test_per=test_per, SHUFFLE_BUFFER_SIZE=SHUFFLE_BUFFER_SIZE)


model_obj = ConvLSTM((DEPTH, WIDTH, HEIGHT, 1), weight_decay=WEIGHT_DECAY, checkpoint_dir=CHECKPOINT_DIR,
                     is_trainng=False, name=NAME, batch_size=BATCH_SIZE, log_dir=LOG_DIR, drop_out_score=DROP_OUT_SCORE, model_path=MODEL_PATH)

model_obj.load_latest_checkpoint()

model_obj.compile_adam(learning_rate=LEARNING_RATE)
# call_backs = model_obj.call_back()
# Eval_results = model_obj.model.evaluate(x=test_dataset)
# print('EVAL_RESULTS: ', Eval_results)

labels = []
predictions = []
print('This is the value:', int(test_dataset_size / BATCH_SIZE))
counter = 0
for images, label in train_dataset.take(1077):  # only take first element of dataset
    array_pre = model_obj.model.predict(images)
    array_lab = label
    if counter%50 == 0:
        print('counter: ', counter)
    counter += 1
    for i in range(array_lab.shape[0]):
        predictions.append(array_pre[i,:])
        labels.append(array_lab[i,:])

pre = np.stack(predictions, axis=0)
lab = np.stack(labels, axis=0)

pre_tensor = tf.argmax(pre, axis=1)
lab_tensor = tf.argmax(lab, axis=1)
k = tf.math.confusion_matrix(lab_tensor, pre_tensor, dtype=tf.dtypes.int32).numpy()
print(k)
