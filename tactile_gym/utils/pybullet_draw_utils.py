import numpy as np
import pybullet as p


def plot_vector(start_point, vector):

    line_width = 2
    line_life_time = 0.1  # 0 for inf
    line_color = (255, 0, 0)

    line_mag = np.linalg.norm(vector)
    line_direction = vector
    line_end_point = start_point + (line_mag * line_direction)

    p.addUserDebugLine(start_point, line_end_point, line_color, line_width, line_life_time)


def plot_line(pos1, pos2, line_life_time=0.1):

    line_width = 2
    line_life_time = 0.1  # 0 for inf
    line_color = (0, 255, 0)

    p.addUserDebugLine(pos1, pos2, line_color, line_width, line_life_time)
