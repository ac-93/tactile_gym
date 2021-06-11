import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import time
import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

from tactile_gym.sb3_helpers.rl_plot_utils import plot_train_and_eval


class SimplePlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    :param verbose: (int)
    """

    def __init__(self, log_dir, total_timesteps, verbose=1):
        super(SimplePlottingCallback, self).__init__(verbose)
        self._plot = None
        self._log_dir = log_dir
        self._total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # get the monitor's data
        x, y = ts2xy(load_results(self._log_dir), "timesteps")
        if self._plot is None:  # make the plot
            plt.ion()
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            (line,) = ax.plot(x, y)
            self._plot = (line, ax, fig)
            plt.show()
        else:  # update and rescale the plot
            self._plot[0].set_data(x, y)
            self._plot[-2].relim()
            self._plot[-2].set_xlim([self._total_timesteps * -0.02, self._total_timesteps * 1.02])
            self._plot[-2].autoscale_view(True, True, True)
            self._plot[-1].canvas.draw()


class FullPlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    :param verbose: (int)
    """

    def __init__(self, log_dir, total_timesteps, verbose=1):
        super(FullPlottingCallback, self).__init__(verbose)
        self._plot = None
        self._log_dir = log_dir
        self._total_timesteps = total_timesteps

    def _on_training_start(self) -> None:

        # generate fake data for initial plot
        plt.ion()
        self.fig, self.axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        self.fig.tight_layout(pad=3.0)
        plt.show()
        plt.pause(0.01)

    def _on_step(self) -> bool:

        for ax in self.axs.flat:
            ax.clear()

        plot_train_and_eval(self._log_dir, fig=self.fig, axs=self.axs)
        for ax in self.axs.flat:
            ax.relim()
            ax.set_xlim([self._total_timesteps * -0.02, self._total_timesteps * 1.02])
            ax.autoscale_view(True, True, True)

        self.fig.canvas.draw()

        self.fig.savefig(
            os.path.join(self._log_dir, "learning_curves.png"),
            dpi=320,
            pad_inches=0.01,
            bbox_inches="tight",
        )

    def _on_training_end(self) -> None:
        plt.close(self.fig)


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
