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

    def metrics(self, val: bool = True) -> None:
        """Plots the progress of the all the metrics used.

        Args:
            val (bool, optional): Plost the metric of the validation set. Defaults to True.
        """
        metrics = [x for x in self.history.keys() if x !=
                   "loss" and x != "val_loss"]
        train_metrics = [x for x in metrics if not x.startswith("val")]

        # if val, then make sure "val_metric" is present
        # Assuming the history.keys() contains loss and metrics
        if val:
            for m in train_metrics:
                if not "val_" + m in metrics:
                    raise KeyError(
                        "Validation Metrics not found in history dict.")

        n_rows = 1
        n_cols = len(train_metrics)
        fig = plt.figure(figsize=(n_cols*6, n_rows*4))
        for i in range(n_rows*n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            ax.plot(self.history[train_metrics[i]])

            ax.set_title("Model " + train_metrics[i].capitalize())
            ax.set_xlabel('epoch')
            ax.set_ylabel(train_metrics[i])

            if val:
                ax.plot(self.history["val_" + train_metrics[i]])
                ax.legend(['train', 'val'], loc='lower right')
            else:
                ax.legend(['train'], loc='lower right')

    def single_metric(self, metric: str = 'accuracy', val: bool = True, show_optimal: bool = False) -> None:
        """Plots the progress of the metric (such as accuracy, recall).

        Args:
            metric (str, optional): the metric to plot. Defaults to "accuracy".
            val (bool, optional): Plots the metric of the validation set. Defaults to True.
            show_optimal (bool, optional): Draws a vertical dotted line at the optimal val_metric; works
            only if 'val' is True. Defaults to False.
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


if __name__ == "__main__":
    history = np.load('history.npy', allow_pickle='TRUE').item()

    hist_plotter = HistoryPlotter(history=history)
    hist_plotter.metrics()
