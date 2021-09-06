from math import e
from typing import List, Tuple
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

    def _single_axes(self, ax, key: str, val: bool = True) -> None:
        """Given the axes object, plot a single variable, if 'val' is True, plot the "val_" variable.

        Args:
            ax (AxesSubplot): Axes Object obtained from fig.add_plot()
            key (str): The variable to plot
            val (bool, optional): Whether to plot validation. Defaults to True.
        """
        ax.plot(self.history[key])
        title = key.split("_")

        # Join if more than 1 element
        if len(title) > 1:
            # .title() is like .capitalize() but does it for an entire sentence
            title = " ".join(title).title()
        else:
            title = title[0].capitalize()

        ax.set_title("Model " + title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)

        if val:
            ax.plot(self.history["val_" + key])
            ax.legend(['train', 'val'], loc='best')
        else:
            ax.legend(['train'], loc='best')

    def _single_fig(self, keys: List[str], n_rows: int, n_cols: int, figsize: Tuple[int], val: bool = True,) -> None:
        """Creates a fig according to the number of items in keys.

        Args:
            keys (List[str]): Contains the list of various variables to plot.
            n_rows (int): Figure rows
            n_cols (int): Figure cols
            figsize (Tuple[int]): The plots size
            val (bool, optional): Whether to plot validation. Defaults to True.
        """

        fig = plt.figure(figsize=figsize)
        for i in range(n_rows*n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            self._single_axes(ax=ax, key=keys[i], val=val)

    def loss(self, val: bool = True, show_optimal: bool = False, figsize: Tuple[int] = (6, 4)) -> None:
        """Plots the train_loss and val_loss (if specified)

        Args:
            val (bool, optional): Plots val_loss
            show_optimal (bool, optional): Draws a vertical dotted line at the lowest val_loss; works
            only if 'val' is True
            figsize (Tuple[int], optional): The figure's size. Defaults to (6,4)
        """
        keys = ['loss']
        self._single_fig(keys=keys,
                         n_rows=1, n_cols=1,
                         figsize=figsize, val=val)
        plt.show()

        # # Old show_optimal code, implement later
        # if show_optimal:
        #         optimal_epoch = np.argmin(self.history['val_loss'])
        #         max_loss = max(np.max(self.history['loss']),
        #                        np.max(self.history['val_loss'])
        #                        )

        #         plt.plot([optimal_epoch, optimal_epoch],
        #                  [0, max_loss],
        #                  'g:',
        #                  lw=2)

    def single_metric(self, metric: str = 'accuracy', val: bool = True, show_optimal: bool = False, figsize: Tuple[int] = (6, 4)) -> None:
        """Plots the progress of the metric (such as accuracy, recall).

        Args:
            metric (str, optional): the metric to plot. Defaults to "accuracy".
            val (bool, optional): Plots the metric of the validation set. Defaults to True.
            show_optimal (bool, optional): Draws a vertical dotted line at the optimal val_metric; works
            only if 'val' is True. Defaults to False
            figsize (Tuple[int], optional): The figure's size. Defaults to (6,4)
        """
        keys = ['loss']
        self._single_fig(keys=[metric],
                         n_rows=1, n_cols=1,
                         figsize=figsize, val=val)
        plt.show()

        # if show_optimal:
        #     # currently works for metrics where high values are good.
        #     optimal_epoch = np.argmax(self.history['val_'+metric])
        #     min_metric = min(np.min(self.history[metric]),
        #                         np.min(self.history['val_'+metric])
        #                         )
        #     max_val_metric = np.max(self.history['val_'+metric])

        #     # vertical line
        #     plt.plot([optimal_epoch, optimal_epoch],
        #                 [min_metric, max_val_metric],
        #                 'g:',
        #                 lw=2)
        #     # horizontal line
        #     plt.plot([0, optimal_epoch],
        #                 [max_val_metric, max_val_metric],
        #                 'g:',
        #                 lw=2)

        #     # set marker on the optimal point
        #     # plt.plot([optimal_epoch],
        #     #          [max_val_metric],
        #     #          marker='o',
        #     #          markersize=5,
        #     #          color='green')

    def _process_row_col(self, row_col: List[int], length: int) -> Tuple[List[int], Tuple[int]]:
        """Given the rows and cols of the plot, if any element == 0, set it to "length/(row_cols != 0)".
        I.e., if row_col=[1,0] and length = 6 then we get a plot with [1,6]
        Also returns the appropriate figsize of the plot

        Args: 
            row_col (List[int]): The user given
            length (int): Number of subplots

        Returns:
            Tuple[List[int], Tuple[int]]: Modified row_cols and the figsize
        """
        if row_col[0] == 0 and row_col[1] == 0:
            raise ValueError("Only one value of row_col list can be 0")
        # many rows
        if row_col[0] == 0:
            row_col[0] = int(length/row_col[1])
            # figsize is the (width, length) of the plot; original - (n_cols*6, n_rows*4)
            # so to scale width, we need n_cols
            # to scale length, we need n_rows
            figsize = (row_col[1]*4, row_col[0]*5)  # make length larger

        # many cols
        elif row_col[1] == 0:
            row_col[1] = int(length/row_col[0])
            figsize = (row_col[1]*6, row_col[0]*4)  # make width larger

        return row_col, figsize

    def metrics(self, row_col: List[int], val: bool = True, figsize: Tuple[int] = None) -> None:
        """Plots the progress of the all the metrics used.

        Args:
            row_col (list): The rows and cols in the plot. If any val=0, we take the 'len(metrics)/(rows_cols!=0)'.
                            I.e., if row_col=[1,0] and len(metrics) = 6 then we get a plot with [1,6]
            val (bool, optional): Plost the metric of the validation set. Defaults to True.
            figsize (Tuple[int], optional): The figure's size, autognerated if not provided. Defaults to None
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

        row_col, auto_figsize = self._process_row_col(
            row_col=row_col,
            length=len(train_metrics)
        )
        # use the generated figsize if not provided.
        if figsize is None:
            figsize = auto_figsize

        self._single_fig(keys=train_metrics,
                         n_rows=row_col[0], n_cols=row_col[1],
                         figsize=figsize, val=val)
        plt.show()

    def all(self, row_col: list, val: bool = True, figsize: Tuple[int] = None) -> None:
        """Plots the progress of all the items tracked in history.history

            Args:
                row_col (list): The rows and cols in the plot. If any val=0, we take the 'len(metrics)/(rows_cols!=0)'.
                                I.e., if row_col=[1,0] and len(metrics) = 6 then we get a plot with [1,6]
                val (bool, optional): Plost the metric of the validation set. Defaults to True.
                figsize (Tuple[int], optional): The figure's size, autognerated if not provided. Defaults to None
        """
        metrics = [x for x in self.history.keys()]
        train_metrics = [x for x in metrics if not x.startswith("val")]

        # Make sure val_ items are present
        if val:
            for m in train_metrics:
                if not "val_" + m in metrics:
                    raise KeyError(
                        "Validation Metrics not found in history dict.")

        row_col, auto_figsize = self._process_row_col(
            row_col=row_col,
            length=len(train_metrics)
        )

        # use the generated figsize if not provided.
        if figsize is None:
            figsize = auto_figsize

        self._single_fig(keys=train_metrics,
                         n_rows=row_col[0], n_cols=row_col[1],
                         figsize=figsize, val=val)
        plt.show()


if __name__ == "__main__":
    history = np.load('history.npy', allow_pickle='TRUE').item()

    hist_plotter = HistoryPlotter(history=history)
