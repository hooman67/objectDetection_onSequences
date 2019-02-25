import os
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.callbacks import Callback, ProgbarLogger, ModelCheckpoint, TensorBoard


class ValOnlyProgbarLogger(ProgbarLogger):
    def __init__(self, verbose, count_mode='samples'):
        # Ignore the `verbose` argument specified in `fit()` and pass
        # `count_mode` upstream
        self.verbose = verbose
        super(ValOnlyProgbarLogger, self).__init__(count_mode)

    def on_train_begin(self, logs=None):
        # filter out the training metrics
        self.params['metrics'] = [m for m in self.params['metrics'] if\
                m.startswith('val_') or m.startswith('train') or m.startswith('loss')]
        self.epochs = self.params['epochs']


class EvaluateCallback(Callback):
    def __init__(self, generator, eval_function):
        self.generator = generator
        self.evaluate = eval_function

    def on_epoch_end(self, epoch, logs=None):
        print("Computing mAP on validation set...")
        average_precisions = self.evaluate(self.generator)
        print("Average precision: ", average_precisions)
        print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))         


class DecayLR(Callback):
    ''' n_epoch = no. of epochs after decay should happen. '''  
    def __init__(self, n_epoch_1, n_epoch_2, decay=0.2):
        super(DecayLR, self).__init__()
        self.n_epoch_1 = n_epoch_1
        self.n_epoch_2 = n_epoch_2
        self.decay = decay

    def on_epoch_begin(self, epoch, logs={}):
        old_lr = K.eval(self.model.optimizer.lr)
        if epoch > 1 and (epoch % self.n_epoch_1 == 0 or epoch % self.n_epoch_2 == 0):
            new_lr = self.decay * old_lr
            print("Decreasing lr from %.6f to %.6f." % (old_lr, new_lr))
            K.set_value(self.model.optimizer.lr, new_lr)


class MultiGPUCheckpoint(ModelCheckpoint):
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model


# Writen by HS
class TrainValTensorBoard_HS(TensorBoard):

    def __init__(self, log_dir,**kwargs):

        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard_HS, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard_HS, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard_HS, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard_HS, self).on_train_end(logs)
        self.val_writer.close()
