import random
import taichi as ti
import time

ti.init(arch=ti.gpu)

dump_image = False
max_num_particles = 256

run_explicit_euler = False
# From 1 to 4.
rk_number = 1
num_jacobi_iteration = 20

"""
dt = ti.var(ti.f32, shape=())
num_particles = ti.var(ti.i32, shape=())
spring_stiffness = ti.var(ti.f32, shape=())
energy = ti.var(ti.f32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())
total_r = ti.var(ti.f32, shape=())

"""
dt = ti.field(dtype=ti.f32, shape=())
num_particles = ti.field(dtype=ti.i32, shape=())
spring_stiffness = ti.field(dtype=ti.f32, shape=())
energy = ti.field(dtype=ti.f32, shape=())
paused = ti.field(dtype=ti.i32, shape=())
damping = ti.field(dtype=ti.f32, shape=())
total_r = ti.field(dtype=ti.f32, shape=())

particle_mass = 1.0
bottom_y = 0.05

x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
vec = ti.Vector.field(2, dtype=ti.f32, shape=2 * max_num_particles)
k1 = ti.Vector.field(2, dtype=ti.f32, shape=2 * max_num_particles)
k2 = ti.Vector.field(2, dtype=ti.f32, shape=2 * max_num_particles)
k3 = ti.Vector.field(2, dtype=ti.f32, shape=2 * max_num_particles)
k4 = ti.Vector.field(2, dtype=ti.f32, shape=2 * max_num_particles)
temp = ti.Vector.field(2, dtype=ti.f32, shape=2 * max_num_particles)

fake_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
# grad[i, j] is gradient of f_i over x_j.
grad = ti.Vector.field(2, dtype=ti.f32, shape=(max_num_particles, max_num_particles), needs_grad=True)

linear_A = ti.field(dtype=ti.f32, shape=(2 * max_num_particles, 2 * max_num_particles))
linear_b = ti.field(dtype=ti.f32, shape=2 * max_num_particles)
linear_r = ti.field(dtype=ti.f32, shape=2 * max_num_particles)
linear_X = ti.field(dtype=ti.f32, shape=2 * max_num_particles)
linear_temp_X = ti.field(dtype=ti.f32, shape=2 * max_num_particles)

# rest_length[i, j] = 0 means i and j are not connected
rest_length = ti.field(ti.f32, shape=(max_num_particles, max_num_particles))

connection_radius = 0.15
gravity = [0, -9.8]

moving_step_latency = 0.0
moving_energy_rate = 0.0

energy_buffer_index = 0
energy_buffer = [1000.0] * 100


def get_moving_average(old, new):
    return old * 0.99 + new * 0.01


@ti.func
def force(x, n, i):
    total_force = ti.Vector(gravity) * particle_mass
    for j in range(n):
        if rest_length[i, j] != 0:
            x_ij = x[i] - x[j]
            total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
    return total_force


@ti.func
def fn(vec, n, i):
    r = ti.Vector([0.0, 0.0])
    if i < n:
        # x
        r = vec[i + n]
    else:
        # v
        r = force(vec, n, i - n) / particle_mass
    return r


@ti.func
def explicit_euler(vec, n):
    for i in range(2 * n):
        k1[i] = fn(vec, n, i)
    for i in range(2 * n):
        vec[i] += k1[i] * dt[None]

@ti.func
def RK2(vec, n):
    for i in range(2 * n):
        k1[i] = fn(vec, n, i)
    for i in range(2 * n):
        temp[i] = vec[i] + k1[i] * dt[None]
    for i in range(2 * n):
        k2[i] = fn(temp, n, i)
    for i in range(2 * n):
        vec[i] += 0.5 * dt[None] * (k1[i] + k2[i])


@ti.func
def RK3(vec, n):
    for i in range(2 * n):
        k1[i] = fn(vec, n, i)
    for i in range(2 * n):
        temp[i] = vec[i] + 0.5 * k1[i] * dt[None]
    for i in range(2 * n):
        k2[i] = fn(temp, n, i)
    for i in range(2 * n):
        temp[i] = vec[i] - dt[None] * k1[i] + 2.0 * dt[None] * k2[i]
    for i in range(2 * n):
        k3[i] = fn(temp, n, i)
    for i in range(2 * n):
        vec[i] += 1.0 / 6.0 * dt[None] * (k1[i] + 4 * k2[i] + k3[i])


@ti.func
def RK4(vec, n):
    for i in range(2 * n):
        k1[i] = fn(vec, n, i)
    for i in range(2 * n):
        temp[i] = vec[i] + 0.5 * k1[i] * dt[None]
    for i in range(2 * n):
        k2[i] = fn(temp, n, i)
    for i in range(2 * n):
        temp[i] = vec[i] + 0.5 * k2[i] * dt[None]
    for i in range(2 * n):
        k3[i] = fn(temp, n, i)
    for i in range(2 * n):
        temp[i] = vec[i] + dt[None] * k3[i]
    for i in range(2 * n):
        k4[i] = fn(temp, n, i)
    for i in range(2 * n):
        vec[i] += 1.0 / 6.0 * dt[None] * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@ti.kernel
def pre_update():
    # Compute force and new velocity
    n = num_particles[None]
    for i in range(n):
        v[i] *= ti.exp(-dt[None] * damping[None])  # damping


@ti.kernel
def post_update():
    n = num_particles[None]
    # Collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = -v[i].y

    energy[None] = 0.0
    for i in range(n):
        r = ti.Vector([0.0])
        r[0] += -gravity[1] * (x[i].y - bottom_y)
        r[0] += 0.5 * particle_mass * (v[i].x * v[i].x + v[i].y * v[i].y)
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = vec[i] - vec[j]
                s = ti.Vector([x_ij.norm() - rest_length[i, j]])
                # A spring is counted twice. So multiply by another 0.5.
                r[0] += 0.5 * 0.5 * spring_stiffness[None] * s[0] * s[0]
        energy[None] += r[0]


@ti.kernel
def explicit_substep():
    n = num_particles[None]

    # pack
    for i in range(n):
        vec[i] = x[i]
        vec[i + n] = v[i]

    if ti.static(rk_number == 1):
        explicit_euler(vec, n)
    if ti.static(rk_number == 2):
        RK2(vec, n)
    if ti.static(rk_number == 3):
        RK3(vec, n)
    if ti.static(rk_number == 4):
        RK4(vec, n)

    # unpack
    for i in range(n):
        x[i] = vec[i]
        v[i] = vec[i + n]


@ti.kernel
def pre_implicit_forward():
    n = num_particles[None]
    for i, j in ti.ndrange(n, n):
        grad[j, i] = x[i]
        linear_A[i, j] = 0.0

@ti.kernel
def reset_loss():
    fake_loss[None] = 0.0


@ti.kernel
def implicit_forward_x(n: ti.i32):
    for i, j in ti.ndrange(n, n):
        if rest_length[i, j] != 0:
            x_ij = x[j] - grad[j, i]
            fake_loss[None] += ((x_ij.norm(eps=1e-5) - rest_length[i, j]) * x_ij.normalized(eps=1e-5)).x
            x_ij = grad[i, i] - x[j]
            fake_loss[None] += ((x_ij.norm(eps=1e-5) - rest_length[i, j]) * x_ij.normalized(eps=1e-5)).x


@ti.kernel
def implicit_forward_y(n: ti.i32):
    for i, j in ti.ndrange(n, n):
        if rest_length[i, j] != 0:
            x_ij = x[j] - grad[j, i]
            fake_loss[None] += ((x_ij.norm(eps=1e-5) - rest_length[i, j]) * x_ij.normalized(eps=1e-5)).y
            x_ij = grad[i, i] - x[j]
            fake_loss[None] += ((x_ij.norm(eps=1e-5) - rest_length[i, j]) * x_ij.normalized(eps=1e-5)).y


@ti.kernel
def post_implicit_forward_x():
    n = num_particles[None]
    for i, j in ti.ndrange(n, n):
        linear_A[2 * i, 2 * j] = -spring_stiffness[None] * grad.grad[i, j].x
        linear_A[2 * i + 1, 2 * j] = -spring_stiffness[None] * grad.grad[i, j].y


@ti.kernel
def post_implicit_forward_y():
    n = num_particles[None]
    for i, j in ti.ndrange(n, n):
        linear_A[2 * i, 2 * j + 1] = -spring_stiffness[None] * grad.grad[i, j].x
        linear_A[2 * i + 1, 2 * j + 1] = -spring_stiffness[None] * grad.grad[i, j].y


@ti.kernel
def compute_linear_equation_A_and_b():
    for i, j in ti.ndrange(2 * n, 2 * n):
        if i == j:
            linear_A[i, j] = 1.0 - dt[None] * dt[None] / particle_mass * linear_A[i, j]
        else:
            linear_A[i, j] = -dt[None] * dt[None] / particle_mass * linear_A[i, j]

    for i in range(n):
        temp[i] = v[i] + force(x, n, i) / particle_mass * dt[None]
        linear_b[2 * i] = temp[i].x
        linear_b[2 * i + 1] = temp[i].y


@ti.kernel
def implicit_substep_pre():
    n = num_particles[None]
    for i in range(n):
        linear_X[2 * i] = v[i].x
        linear_X[2 * i + 1] = v[i].y

@ti.kernel
def jacobi_iterate():
    n = num_particles[None]
    total_r[None] = 0.0
    for i in range(2 * n):
        r = linear_b[i]
        for j in range(2 * n):
            if i != j:
                r -= linear_A[i, j] * linear_X[j]
        linear_temp_X[i] = r / linear_A[i, i]
        total_r[None] += ti.abs(r)
    for i in range(2 * n):
        linear_X[i] = linear_temp_X[i]


@ti.kernel
def implicit_substep_post():
    n = num_particles[None]
    for i in range(n):
        v[i].x = linear_X[2 * i]
        v[i].y = linear_X[2 * i + 1]
    for i in range(n):
        x[i] += dt[None] * v[i]


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32):  # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        if dist < connection_radius:
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1


gui = ti.GUI('Mass Spring System', res=(1024, 1024), background_color=0xdddddd)

spring_stiffness[None] = 1000.0
damping[None] = 20.0
dt[None] = 1e-3

# new_particle(0.3, 0.6)
# new_particle(0.3, 0.7)
# new_particle(0.4, 0.7)
random.seed(2020)
for i in range(20):
    new_particle(random.random() * 0.3 + 0.4, random.random() * 0.3 + 0.6)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1])
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                spring_stiffness[None] /= 1.1
            else:
                spring_stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1
        elif e.key == 't':
            if gui.is_pressed('Shift'):
                dt[None] /= 1.1
            else:
                dt[None] *= 1.1

    if not paused[None]:
        for step in range(10):
            tt = time.time()
            ss = 0.0
            pre_update()
            if run_explicit_euler:
                explicit_substep()
            else:
                # Get gradient
                pre_implicit_forward()
                n = num_particles[None]

                ss = time.time()
                # TODO: compute gradients of x and y in one pass.
                reset_loss()
                with ti.Tape(loss=fake_loss):
                    implicit_forward_x(n)
                post_implicit_forward_x()
                reset_loss()
                with ti.Tape(loss=fake_loss):
                    implicit_forward_y(n)
                post_implicit_forward_y()
                ss = time.time() - ss

                compute_linear_equation_A_and_b()

                implicit_substep_pre()
                for i in range(num_jacobi_iteration):
                    jacobi_iterate()
                implicit_substep_post()

            post_update()
            tt = time.time() - tt - ss
            moving_step_latency = get_moving_average(moving_step_latency, tt * 1000000)

            ee = energy[None] + 1e-8
            moving_energy_rate = get_moving_average(moving_energy_rate, ee / energy_buffer[energy_buffer_index % 100])
            energy_buffer[energy_buffer_index % 100] = ee
            energy_buffer_index += 1

    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)

    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)

    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.text(content=f'Energy change rate {moving_energy_rate:.2f}', pos=(0, 0.8), color=0x0)
    gui.text(content=f'Jacob R {total_r[None]:.2f}', pos=(0, 0.75), color=0x0)
    gui.text(content=f'Step latency (microsecond) {moving_step_latency:.2f}', pos=(0, 0.7), color=0x0)
    gui.text(content=f'Dt (second) {dt[None]:.5f}', pos=(0, 0.65), color=0x0)

    if dump_image:
        filename = 'frame_{:05d}.png'.format(energy_buffer_index // 10)   # create filename with suffix png
        print('Frame {} is recorded in {}'.format(energy_buffer_index // 10, filename))
        gui.show(filename)
    else:
        gui.show()