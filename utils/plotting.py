import matplotlib.pyplot as plt
import numpy as np


class HistoryPlotter():
    """Various ways to plot the training history
    """

    def __init__(self, history: dict) -> None:
        """Takes in the history dict

        Args:
            history (dict): The history dictionary, i.e., 'model_history.history'
        """
        self.history = history

    def loss(self, val: bool = True, show_optimal: bool = False) -> None:
        """Plots the train_loss and val_loss (if specified)

        Args:
            val (bool, optional): Plots val_loss
            show_optimal (bool, optional): Draws a vertical dotted line at the lowest val_loss; works
            only if 'val' is True
        """

        plt.plot(self.history["loss"])
        plt.title('Model Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')

        if val:
            plt.plot(self.history["val_loss"])
            plt.legend(['train', 'val'], loc='upper right')

            if show_optimal:
                optimal_epoch = np.argmin(self.history['val_loss'])
                max_loss = max(np.max(self.history['loss']),
                               np.max(self.history['val_loss'])
                               )

                plt.plot([optimal_epoch, optimal_epoch],
                         [0, max_loss],
                         'g:',
                         lw=2)

        else:
            plt.legend(['train'], loc='upper right')
        plt.show()

    def metric(self, metric: str = 'accuracy', val: bool = True, show_optimal: bool = False) -> None:
        """Plots the progress of the metric (such as accuracy, recall).

        Args:
            metric (str, optional): the metric to plot
            val (bool, optional): Plots the metric of the validation set
            show_optimal (bool, optional): Draws a vertical dotted line at the optimal val_metric; works
            only if 'val' is True
        """
        plt.plot(self.history[metric])
        plt.title('Model ' + metric.capitalize())
        plt.xlabel('epoch')
        plt.ylabel(metric)

        if val:
            plt.plot(self.history["val_" + metric])
            plt.legend(['train', 'val'], loc='lower right')

            if show_optimal:
                # currently works for metrics where high values are good.
                optimal_epoch = np.argmax(self.history['val_'+metric])
                min_metric = min(np.min(self.history[metric]),
                                 np.min(self.history['val_'+metric])
                                 )
                max_val_metric = np.max(self.history['val_'+metric])

                # vertical line
                plt.plot([optimal_epoch, optimal_epoch],
                         [min_metric, max_val_metric],
                         'g:',
                         lw=2)
                # horizontal line
                plt.plot([0, optimal_epoch],
                         [max_val_metric, max_val_metric],
                         'g:',
                         lw=2)

                # set marker on the optimal point
                # plt.plot([optimal_epoch],
                #          [max_val_metric],
                #          marker='o',
                #          markersize=5,
                #          color='green')

        else:
            plt.legend(['train'], loc='lower right')
        plt.show()
