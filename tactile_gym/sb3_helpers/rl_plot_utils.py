import matplotlib.pyplot as plt
import numpy as np
import os

from tactile_gym.utils.general_utils import load_json_obj
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter


def plot_error_band(
    axs,
    x_data,
    y_data,
    min=None,
    max=None,
    x_label=None,
    y_label=None,
    title=None,
    colour=None,
    error_band=False,
    label=None,
):

    (line,) = axs.plot(x_data, y_data, color=colour, alpha=0.75, label=label)

    if error_band:
        fill = axs.fill_between(x_data, min, max, color=colour, alpha=0.25)

    axs.set(xlabel=x_label, ylabel=y_label)
    axs.set_xlim([0, np.max(x_data)])
    # axs.set_ylim([np.min(min), np.max(max)])
    axs.set_title(title)

    for item in [axs.title]:
        item.set_fontsize(10)

    for item in [axs.xaxis.label, axs.yaxis.label]:
        item.set_fontsize(9)

    for item in axs.get_xticklabels() + axs.get_yticklabels():
        item.set_fontsize(8)

    if error_band:
        return line, fill
    else:
        return line


def load_train_results(saved_model_dir):
    results = load_results(saved_model_dir)

    timesteps, _ = ts2xy(results, "timesteps")
    rewards = np.array(results["r"])
    ep_lengths = np.array(results["l"])

    return timesteps, rewards, ep_lengths


def plot_train_results(saved_model_dir, window_size=50, show_plot=False):

    timesteps, reward, ep_lengths = load_train_results(saved_model_dir)

    # if the data size is less than the window size return zeros
    if timesteps.shape[0] < window_size:
        print("Cannot apply windows of size {} to data of size {}".format(window_size, timesteps.shape[0]))
        return None, None

    x_r, mean_rew = results_plotter.window_func(timesteps, reward, window_size, np.mean)
    x_l, mean_ep_len = results_plotter.window_func(timesteps, ep_lengths, window_size, np.mean)

    _, max_rew = results_plotter.window_func(timesteps, reward, window_size, np.max)
    _, min_rew = results_plotter.window_func(timesteps, reward, window_size, np.min)

    _, max_ep_len = results_plotter.window_func(timesteps, ep_lengths, window_size, np.max)
    _, min_ep_len = results_plotter.window_func(timesteps, ep_lengths, window_size, np.min)

    if show_plot:

        fig, axs = plt.subplots(nrows=2, ncols=1)
        fig.tight_layout(pad=3.0)

        plot_error_band(
            axs[0],
            x_r,
            mean_rew,
            min_rew,
            max_rew,
            x_label="TimeSteps",
            y_label="AvgTrainEpRew",
            title=None,
            colour="r",
            error_band=True,
        )
        plot_error_band(
            axs[1],
            x_l,
            mean_ep_len,
            min_ep_len,
            max_ep_len,
            x_label="TimeSteps",
            y_label="AvgTrainEpLength",
            title=None,
            colour="b",
            error_band=True,
        )
        plt.show()

    return [x_r, mean_rew, min_rew, max_rew], [x_l, mean_ep_len, min_ep_len, max_ep_len]


def load_eval_results(saved_model_dir):
    eval_data = np.load(os.path.join(saved_model_dir, "trained_models", "evaluations.npz"))
    timesteps = eval_data["timesteps"]
    rewards = eval_data["results"]
    ep_lengths = eval_data["ep_lengths"]
    return timesteps, rewards, ep_lengths


def plot_eval_results(saved_model_dir, show_plot=False):

    # load the data
    try:
        timesteps, rewards, ep_lengths = load_eval_results(saved_model_dir)
    except:
        print("No saved evaluation data")
        return None, None

    # get useful info from data
    mean_rew = np.mean(rewards, axis=1).squeeze()
    min_rew = np.min(rewards, axis=1).squeeze()
    max_rew = np.max(rewards, axis=1).squeeze()

    mean_ep_len = np.mean(ep_lengths, axis=1).squeeze()
    min_ep_len = np.min(ep_lengths, axis=1).squeeze()
    max_ep_len = np.max(ep_lengths, axis=1).squeeze()

    if show_plot:

        fig, axs = plt.subplots(nrows=2, ncols=1)
        fig.tight_layout(pad=3.0)

        plot_error_band(
            axs[0],
            timesteps,
            mean_rew,
            min_rew,
            max_rew,
            x_label="TimeSteps",
            y_label="AvgEvalEpRew",
            title=None,
            colour="r",
            error_band=True,
        )
        plot_error_band(
            axs[1],
            timesteps,
            mean_ep_len,
            min_ep_len,
            max_ep_len,
            x_label="TimeSteps",
            y_label="AvgEvalEpLength",
            title=None,
            colour="b",
            error_band=True,
        )

        plt.show()

    return [timesteps, mean_rew, min_rew, max_rew], [
        timesteps,
        mean_ep_len,
        min_ep_len,
        max_ep_len,
    ]


def plot_train_and_eval(saved_model_dir, fig=None, axs=None, show_plot=False, save_plot=False):

    if fig == None:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        fig.tight_layout(pad=3.0)

    train_rew_data, train_len_data = plot_train_results(saved_model_dir, show_plot=False)
    eval_rew_data, eval_len_data = plot_eval_results(saved_model_dir, show_plot=False)

    if train_rew_data is not None:
        plot_error_band(
            axs[0][0],
            train_rew_data[0],
            train_rew_data[1],
            train_rew_data[2],
            train_rew_data[3],
            x_label="TimeSteps",
            y_label="AvgTrainEpRew",
            title=None,
            colour="r",
            error_band=True,
        )

    if train_len_data is not None:
        plot_error_band(
            axs[1][0],
            train_len_data[0],
            train_len_data[1],
            train_len_data[2],
            train_len_data[3],
            x_label="TimeSteps",
            y_label="AvgTrainEpLength",
            title=None,
            colour="b",
            error_band=True,
        )

    if eval_rew_data is not None:
        plot_error_band(
            axs[0][1],
            eval_rew_data[0],
            eval_rew_data[1],
            eval_rew_data[2],
            eval_rew_data[3],
            x_label="TimeSteps",
            y_label="AvgEvalEpRew",
            title=None,
            colour="r",
            error_band=True,
        )

    if eval_len_data is not None:
        plot_error_band(
            axs[1][1],
            eval_len_data[0],
            eval_len_data[1],
            eval_len_data[2],
            eval_len_data[3],
            x_label="TimeSteps",
            y_label="AvgEvalEpLength",
            title=None,
            colour="b",
            error_band=True,
        )

    if show_plot:
        plt.show()

    if save_plot:
        fig.savefig(
            os.path.join(saved_model_dir, "learning_curves.png"),
            dpi=320,
            pad_inches=0.01,
            bbox_inches="tight",
        )


def plot_obs_stack(obs_stack, n_obs_channel=1):

    n_stack = int(obs_stack.shape[1] / n_obs_channel)
    for i in range(n_stack):

        # setup plot
        num_envs = obs_stack.shape[0]
        n_fig_columns = 4
        n_fig_rows = num_envs // n_fig_columns if num_envs % n_fig_columns == 0 else (num_envs // n_fig_columns) + 1
        fig, axs = plt.subplots(n_fig_rows, n_fig_columns, figsize=(10, 10))
        axs = axs.flat

        for env_id in range(num_envs):

            obs = obs_stack[env_id, ...]

            # include channel depth
            image_list = []
            for j in range(n_obs_channel):
                image = obs[(i * n_obs_channel) + j, ...]
                image_list.append(image)

            if n_obs_channel == 3:
                concat_img = np.dstack(image_list)
            else:
                concat_img = np.hstack(image_list)

            # plot the image
            if n_obs_channel == 1:
                axs[env_id].imshow(concat_img, cmap="gray")
            else:
                axs[env_id].imshow(concat_img)

            axs[env_id].set_title("Env {}, Frame {}".format(env_id, i))

        plt.show(block=True)


def show_stacked_imgs(obs_stack, n_img_channel=3, max_display=16):
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 12))
    n_batch = int(obs_stack.shape[0])
    n_stack = int(obs_stack.shape[1] / n_img_channel)

    for i in range(1, n_stack + 1):

        grid = (
            make_grid(obs_stack[:max_display, (i - 1) * n_img_channel : i * n_img_channel, ...], 4)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        fig.add_subplot(1, n_stack, i)
        plt.xticks([])
        plt.yticks([])
        plt.title("frame " + str(i))
        plt.imshow(grid)

    plt.show(block=True)


if __name__ == "__main__":
    saved_model_dir = "saved_models/edge_follow/ppo/enjoy_oracle"
    plot_train_and_eval(saved_model_dir, show_plot=True)
