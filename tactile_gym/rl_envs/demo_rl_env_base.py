import time
import os
import numpy as np


def demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info=False):
    """
    Control loop for demonstrating an RL environment.
    Use show_gui and show_tactile flags for visualising and controlling the env.
    Use render for more direct info on what the agent will see.
    """
    record = False
    if record:
        import imageio

        render_frames = []
        log_id = env._pb.startStateLogging(
            loggingType=env._pb.STATE_LOGGING_VIDEO_MP4, fileName=os.path.join("example_videos", "gui.mp4")
        )

    # collection loop
    for i in range(num_iter):
        r_sum = 0
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        step = 0

        while not d:

            if show_gui:
                a = []
                for action_id in action_ids:
                    a.append(env._pb.readUserDebugParameter(action_id))
            else:
                a = env.action_space.sample()

            # step the environment
            o, r, d, info = env.step(a)

            if print_info:
                print("")
                print("Step: ", step)
                print("Act:  ", a)
                print("Obs:  ")
                for key, value in o.items():
                    if value is None:
                        print("  ", key, ":", value)
                    else:
                        print("  ", key, ":", value.shape)
                print("Rew:  ", r)
                print("Done: ", d)

            # render visual + tactile observation
            if render:
                render_img = env.render()
                if record:
                    render_frames.append(render_img)

            r_sum += r
            step += 1

            q_key = ord("q")
            r_key = ord("r")
            keys = env._pb.getKeyboardEvents()
            if q_key in keys and keys[q_key] & env._pb.KEY_WAS_TRIGGERED:
                exit()
            elif r_key in keys and keys[r_key] & env._pb.KEY_WAS_TRIGGERED:
                d = True

        print("Total Reward: ", r_sum)

    if record:
        env._pb.stopStateLogging(log_id)
        imageio.mimwrite(os.path.join("example_videos", "render.mp4"), np.stack(render_frames), fps=12)

    env.close()
