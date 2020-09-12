
import tensorflow as tf
import tensorboard
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import pickle5 as pickle
class ConvLSTM():
    def __init__(self, input_shape, weight_decay=0.0001, checkpoint_dir='./', is_trainng=True, name = 'FXCM_LookUp_resume', batch_size = 2, log_dir=None, drop_out_score= 0, model_path=None):
        self._input_shape = input_shape
        self.model_path = model_path
        self.drop_out_score = drop_out_score
        self._weight_decay = weight_decay
        self._depth = input_shape[2]
        self._name = name
        self._checkpoint_dir = checkpoint_dir
        self._is_training = is_trainng
        self.model = self.predictor_model()
        self.log_dir = log_dir
        # self.tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, batch_size=batch_size)
        # self.tb.set_model(self.model)

    def predictor_model(self):
        X = Input(self._input_shape)
        print(X.shape)

        l_2 = tf.keras.layers.ConvLSTM2D(16, 5, strides=(1, 1), padding='same', return_sequences=True, name = 'l_2')(X)
        l_2 = tf.keras.layers.LeakyReLU()(l_2)
        l_1 = tf.keras.layers.ConvLSTM2D(16, 5, strides=(1, 1), padding='same', return_sequences=True, name = 'l_1')(l_2)
        l_1 = tf.keras.layers.LeakyReLU()(l_1)
        l_0 = tf.keras.layers.ConvLSTM2D(16, 5, strides=(1, 1), padding='same', return_sequences=True, name='l_0')(l_1)
        l_0 = tf.keras.layers.LeakyReLU()(l_0)
        # l_0 = tf.keras.layers.BatchNormalization()(l_0)

        l0 = tf.keras.layers.ConvLSTM2D(32, 5, strides=(2, 2), padding='same', return_sequences=True, name = 'l0')(l_0)
        l0 = tf.keras.layers.LeakyReLU()(l0)
        l1 = tf.keras.layers.ConvLSTM2D(32, 5, strides=(1, 1), padding='same', return_sequences=True, name = 'l1')(l0)
        l1 = tf.keras.layers.LeakyReLU()(l1)
        l1_1 = tf.keras.layers.ConvLSTM2D(32, 5, strides=(1, 1), padding='same', return_sequences=True, name='l1_1')(l1)
        l1_1 = tf.keras.layers.LeakyReLU()(l1_1)
        l1_1 = tf.keras.layers.Concatenate()([l1_1, l0])
        l1_1 = tf.keras.layers.BatchNormalization()(l1_1)

        l2_1 = tf.keras.layers.ConvLSTM2D(32, 3, strides=(1, 1), padding='same', return_sequences=True, name = 'l2_1')(l1_1)
        l2_1 = tf.keras.layers.LeakyReLU()(l2_1)
        l2 = tf.keras.layers.ConvLSTM2D(32, 3, strides=(1, 1), padding='same', return_sequences=True, name = 'l2')(l2_1)
        l2 = tf.keras.layers.LeakyReLU()(l2)
        l2_2 = tf.keras.layers.ConvLSTM2D(32, 3, strides=(1, 1), padding='same', return_sequences=True, name='l2_2')(l2)
        l2_2 = tf.keras.layers.LeakyReLU()(l2_2)
        # l2 = tf.keras.layers.BatchNormalization()(l2_2)
        l2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 1, 1))(l2_2)

        l3 = tf.keras.layers.ConvLSTM2D(64, 3, strides=(2, 2), padding='same', return_sequences=True, name = 'l3')(l2)
        l3 = tf.keras.layers.LeakyReLU()(l3)
        l4 = tf.keras.layers.ConvLSTM2D(64, 3, strides=(1, 1), padding='same', return_sequences=True, name = 'l4')(l3)
        l4 = tf.keras.layers.LeakyReLU()(l4)
        l4_1 = tf.keras.layers.ConvLSTM2D(64, 3, strides=(1, 1), padding='same', return_sequences=True, name='l4_1')(l4)
        l4_1 = tf.keras.layers.LeakyReLU()(l4_1)
        l4_2 = tf.keras.layers.Concatenate()([l4_1, l3])
        l4_2 = tf.keras.layers.BatchNormalization()(l4_2)
        l4_2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 1, 1))(l4_2)

        l5 = tf.keras.layers.ConvLSTM2D(64, 3, strides=(2, 1), padding='same', return_sequences=True, name = 'l5')(l4_2)
        l5 = tf.keras.layers.LeakyReLU()(l5)
        l6 = tf.keras.layers.ConvLSTM2D(64, 3, strides=(1, 1), padding='same', return_sequences=True, name='l6')(l5)
        l6 = tf.keras.layers.LeakyReLU()(l6)
        l7 = tf.keras.layers.ConvLSTM2D(64, 3, strides=(1, 1), padding='same', return_sequences=True, name='l7')(l6)
        l7 = tf.keras.layers.LeakyReLU()(l7)
        # l7 = tf.keras.layers.BatchNormalization()(l7)
        l7 = tf.keras.layers.MaxPooling3D(pool_size=(2, 1, 1))(l7)

        l8 = tf.keras.layers.ConvLSTM2D(128, 3, strides=(1, 1), padding='same', return_sequences=True, name='l8')(l7)
        l8 = tf.keras.layers.LeakyReLU()(l8)
        # l9 = tf.keras.layers.ConvLSTM2D(128, 3, strides=(1, 1), padding='same', return_sequences=True, name='l9')(l8)
        # l9 = tf.keras.layers.LeakyReLU()(l9)
        # l10 = tf.keras.layers.ConvLSTM2D(128, 3, strides=(1, 1), padding='same', return_sequences=True, name='l10')(l9)
        # l10 = tf.keras.layers.LeakyReLU()(l10)
        # l10 = tf.keras.layers.Concatenate()([l10, l8])
        l10 = tf.keras.layers.BatchNormalization()(l8)
        # l10 = tf.keras.layers.MaxPooling3D(pool_size=(2, 1, 1))(l10)

        # l12 = tf.keras.layers.ConvLSTM2D(128, 1, strides=(1, 1), padding='same', return_sequences=True, name='l12')(l10)
        # l12 = tf.keras.layers.LeakyReLU()(l12)
        # l13 = tf.keras.layers.ConvLSTM2D(128, 1, strides=(1, 1), padding='same', return_sequences=True, name='l13')(l12)
        # l13 = tf.keras.layers.LeakyReLU()(l13)
        # l14 = tf.keras.layers.ConvLSTM2D(128, 1, strides=(1, 1), padding='same', return_sequences=True, name='l14')(l13)
        # l14 = tf.keras.layers.LeakyReLU()(l14)
        # l14 = tf.keras.layers.BatchNormalization()(l14)

        l18 = tf.squeeze(l10, 3)
        l18 = tf.keras.layers.MaxPooling2D((3,3))(l18)
        l18 = tf.keras.layers.Flatten()(l18)

        l18 = tf.keras.layers.Dropout(self.drop_out_score)(l18)
        l_dense0 = Dense(64, name = 'l_dense0')(l18)
        l_dense0 = tf.keras.layers.LeakyReLU()(l_dense0)
        l_dense1 = Dense(16, name = 'l_dense1')(l_dense0)
        l_dense1 = tf.keras.layers.LeakyReLU()(l_dense1)
        # l_dense1 = tf.keras.layers.Dropout(self.drop_out_score)(l_dense1)
        # l_dense2 = Dense(1024, activation='relu', name = 'l_dense2')(l_dense1)
        # l_dense2 = tf.keras.layers.Dropout(self.drop_out_score)(l_dense2)
        # l_dense3 = Dense(1024, activation='relu', name = 'l_dense3')(l_dense2)
        output = Dense(3, activation=None)(l_dense1)
        if not self._is_training:
            output = tf.keras.layers.Softmax()(output)

        model = Model(X, output, name=self._name)
        print(model.summary())

        return model

    def compile_adam(self,learning_rate=0.01):
        self.opt = Adam(lr=learning_rate)

        # METRICS = [
        #     tf.keras.metrics.TruePositives(name='tp'),
        #     tf.keras.metrics.FalsePositives(name='fp'),
        #     tf.keras.metrics.TrueNegatives(name='tn'),
        #     tf.keras.metrics.FalseNegatives(name='fn'),
        #     tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        #     tf.keras.metrics.Precision(name='precision'),
        #     tf.keras.metrics.Recall(name='recall'),
        #     tf.keras.metrics.AUC(name='auc'),
        # ]

        self.load_latest_checkpoint()
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=self.opt,
                           metrics=['CategoricalAccuracy', 'CategoricalCrossentropy'])

    def fit(self, ds_train, epochs=10, batch_size=16, num_sample=489, testing_data=None, validation_steps=2880, history_loc='./history.p', class_weight=None):
        history = self.model.fit(ds_train,
                                steps_per_epoch=int(num_sample / (batch_size)),
                                verbose=1,
                                epochs=epochs,
                                callbacks=self.call_back(),
                                use_multiprocessing=True,
                                shuffle=True,
                                class_weight=class_weight,
                                validation_data = testing_data,
                                validation_steps = int(validation_steps / (batch_size))
                                )
        # history = self.model.train_on_batch()
        pickle.dump(history.history, open(history_loc, "wb"))


        print('\nhistory dict:', history.history)

    def error(self, y_true, y_pred, al=0.84):
        sm_ssim = tf.image.ssim_multiscale(y_true, y_pred, 1.0)
        print(sm_ssim.shape)
        l1 = tf.keras.losses.MAE(y_true, y_pred)
        loss = al * tf.reduce_mean(sm_ssim) + (1 - al) * l1
        return loss

    def call_back(self):

        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=1, min_delta=0.0001,
                                                         mode='min', )

        modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(self._checkpoint_dir + self._name +'_{epoch}.h5', monitor='val_loss', save_best_only=True,
                                                             mode='min', verbose=1, save_weights_only=True)

        # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, write_graph=True, update_freq='batch')

        callbacks_list = [modelCheckpoint, earlyStopping]

        return callbacks_list

    def save(self, epoch):
        self.model.save(self._checkpoint_dir + '_' + self._name +'_'+ str(epoch) +'.h5')

    def named_logs(self, logs):
        result = {}
        for l in zip(self.model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def train_on_batch(self, batch, training_batch, label_batch):
        logs = self.model.train_on_batch(training_batch, label_batch)
        self.tb.on_epoch_end(batch, self.named_logs([logs]))
        return logs

    def eval_training(self, testing_batch, label_batch):
        temp = self.model.predict(testing_batch)
        return tf.keras.losses.MSE(label_batch, temp)

    # restoring the model at training initializtion
    def load_latest_checkpoint(self):
        # check to see if checkpoint exists
        # latest_checkpoint = tf.train.latest_checkpoint('D:\\Programming\\fxcm\\ConvLSTM\\', latest_filename=self._name)
        if (self.model_path):
            print("restoring from latest checkpoint {}".format(self.model_path))
            self.model.load_weights(self.model_path)
            # self.checkpoint = tf.train.Checkpoint(
            # self.checkpoint.restore(self.model_path)
        else:
            print("No checkpoint found, initiliazing from scratch")

    def get_model(self, latest_checkpoint=None):
        if latest_checkpoint is not None:
            self.model.load_weights(latest_checkpoint)
        return self.model

    def model_predict(self, i):

        self.load_latest_checkpoint()
        pre = self.model.predict(i)
        return pre

