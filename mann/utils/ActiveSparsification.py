import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from .utils import get_custom_objects, mask_model

class ActiveSparsification(Callback):
    """
    Keras-compatible callback object which enables active sparsification, allowing for increased sparsification as models 
    train.    
    """
    def __init__(
        self,
        performance_cutoff,
        performance_measure = 'auto',
        starting_sparsification = None,
        max_sparsification = 99,
        sparsification_rate = 1,
        sparsification_patience = 10,
        stopping_delta = 0.01,
        stopping_patience = 5,
        restore_best_weights = True,
        verbose = 1
    ):
        """
        Parameters
        ----------
        performance_cutoff : float
            The cutoff value that the performance measure must "beat" in order to iteratively sparsify
        performance_measure : str (default 'auto')
            The performance measure that is used in conjunction with `performance_cutoff`
        starting_sparsification : int or None (default None)
            The starting sparsification that the model has already been sparsified to. If `None`, then defaults to 0
        max_sparsification : int (default 99)
            The maximum sparsification allowed
        sparsification_rate : int (default 1)
            The increase in sparsification that occurs when model performance beats the performance cutoff
        sparsification_patience : int (default 10)
            The number of epochs the model is allowed to train for without beating the performance measure before stopping sparsification
        stopping_delta : float (default 0.01)
            The performance improvement that must be seen when pruning has stopped and early stopping is being considered
        stopping_patience : int (default 5)
            The number of epochs the model is allowed to train for without `stopping_delta` improvement before stopping training
        restore_best_weights : bool (default True)
            Whether to restore model best weights after training
        verbose : int or bool (default 1)
            Verbosity level for logging. 

        Notes
        -----
        - If `performance_measure` is 'auto', defaults to the following measures, in order: 'val_accuracy', 'val_loss', 'accuracy', 'loss'
        - If `performance_measure` defaults to any accuracy value, then `performance_cutoff` represents the minimum value that must be 
          beaten.  If `performance_measure` defaults to any loss value, then `performance_cutoff` represents the maximum value that must 
          be beaten
        """
        super(Callback, self).__init__()
        self.performance_cutoff = performance_cutoff
        self.performance_measure = performance_measure
        self.starting_sparsification = starting_sparsification
        self.max_sparsification = max_sparsification
        self.sparsification_rate = sparsification_rate
        self.sparsification_patience = sparsification_patience
        self.stopping_delta = stopping_delta
        self.stopping_patience = stopping_patience
        self.restore_best_weights = restore_best_weights
        self.verbose = int(verbose)

    def on_train_begin(self, logs = None):
        self.prune_wait = 0
        self.stop_wait = 0

        self.best_weights = self.model.get_weights()
        self.best = None
        self.pruning = True
        self.sparsification = self.starting_sparsification if self.starting_sparsification is not None else 0
    
    def on_epoch_end(self, epoch, logs = None):
        if self.performance_measure == 'auto':
            if 'val_accuracy' in logs.keys():
                self.performance_measure = 'val_accuracy'
            elif 'val_loss' in logs.keys():
                self.performance_measure = 'val_loss'
            elif 'accuracy' in logs.keys():
                self.performance_measure = 'accuracy'
            else:
                self.performance_measure = 'loss'
            if self.verbose:
                print(f'Performance measure set to {self.performance_measure}')
        
        performance = logs[self.performance_measure]
        if self.best is None:
            self.best = performance

        if 'accuracy' in self.performance_measure:
            if self.pruning:
                if performance >= self.performance_cutoff:
                    self.best_weights = self.model.get_weights()
                    self.best = performance
                    if self.sparsification + self.sparsification_rate > self.max_sparsification:
                        print('Model cannot be sparsified further due to max sparsification parameter')
                        self.pruning = False
                    else:
                        self._sparsify_model(self.sparsification + self.sparsification_rate)
                        self.sparsification = self.sparsification + self.sparsification_rate
                        self.prune_wait = 0
                        if self.verbose:
                            print(f'Model performance reached {round(performance, 2)}, sparsifying to {self.sparsification}')
                else:
                    self.prune_wait += 1
                    if self.verbose:
                        print(f'Model performance has not reached pruning threshold for {self.prune_wait} epoch(s)')
                    if self.prune_wait >= self.sparsification_patience:
                        self.pruning = False
                        self.model.set_weights(self.best_weights)
                        if self.verbose:
                            print(f'Model performance has not reached pruning threshold for {self.prune_wait} epochs, reverting to {self.sparsification - self.sparsification_rate} sparsification and beginning early stopping')
            else:
                if performance >= self.best + self.stopping_delta:
                    self.best_weights = self.model.get_weights()
                    self.best = performance
                    self.stop_wait = 0
                    if self.verbose:
                        print(f'Model performance improved to {round(self.best, 2)}')
                else:
                    self.stop_wait += 1
                    if self.verbose:
                        print(f'Early stopping performance has not met threshold for {self.stop_wait} epochs')
                    if self.stop_wait >= self.stopping_patience:
                        if self.restore_best_weights:
                            self.model.set_weights(self.best_weights)
                        if self.verbose:
                            print('Model performance has not met early stopping criteria. Stopping training')
                        self.model.stop_training = True
        
        else:
            if self.pruning:
                if performance <= self.performance_cutoff:
                    self.best_weights = self.model.get_weights()
                    self.best = performance
                    if self.sparsification + self.sparsification_rate > self.max_sparsification:
                        print('Model cannot be sparsified further due to max sparsification parameter')
                        self.pruning = False
                    else:
                        self._sparsify_model(self.sparsification + self.sparsification_rate)
                        self.sparsification = self.sparsification + self.sparsification_rate
                        self.prune_wait = 0
                        if self.verbose:
                            print(f'Model performance reached {round(performance, 2)}, sparsifying to {self.sparsification}')
                else:
                    self.prune_wait += 1
                    if self.verbose:
                        print(f'Model performance has not reached pruning threshold for {self.prune_wait} epoch(s)')
                    if self.prune_wait >= self.sparsification_patience:
                        self.pruning = False
                        self.model.set_weights(self.best_weights)
                        if self.verbose:
                            print(f'Model performance has not reached pruning threshold for {self.prune_wait} epochs, reverting to {self.sparsification - self.sparsification_rate} sparsification and beginning early stopping')
            else:
                if performance <= self.best - self.stopping_delta:
                    self.best_weights = self.model.get_weights()
                    self.best = performance
                    self.stop_wait = 0
                    if self.verbose:
                        print(f'Model performance improved to {round(self.best, 2)}')
                else:
                    self.stop_wait += 1
                    if self.verbose:
                        print(f'Early stopping performance has not met threshold for {self.stop_wait} epochs')
                    if self.stop_wait >= self.stopping_patience:
                        if self.restore_best_weights:
                            self.model.set_weights(self.best_weights)
                        if self.verbose:
                            print('Model performance has not met early stopping criteria. Stopping training')
                        self.model.stop_training = True

    def _sparsify_model(self, percentage):
        """Function to sparsify the model"""
        try:
            new_model = tf.keras.models.Model.from_config(self.model.get_config(), custom_objects = get_custom_objects())
        except:
            new_model = tf.keras.models.Sequential.from_config(self.model.get_config(), custom_objects = get_custom_objects())
        new_model.set_weights(self.model.get_weights())
        self.model.set_weights(
            mask_model(
                new_model,
                percentage,
                method = 'magnitude'
            ).get_weights()
        )
