import taichi as ti

ti.init()

"""
@ti.kernel
def p() -> ti.f32:
    print(42)
    return 40 + 2


print(p())
"""


@ti.func
def taichi_logo(pos: ti.template(), scale: float = 1 / 1.11):
    p = (pos - 0.5) / scale + 0.5
    ret = -1
    if not (p - 0.50).norm_sqr() <= 0.52**2:
        if ret == -1:
            ret = 0
    if not (p - 0.50).norm_sqr() <= 0.495**2:
        if ret == -1:
            ret = 1
    if (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.08**2:
        if ret == -1:
            ret = 1
    if (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.08**2:
        if ret == -1:
            ret = 0
    # bottom white ball
    if (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.25**2:
        if ret == -1:
            ret = 0
    # top balck ball
    if (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.25**2:
        if ret == -1:
            ret = 1
    if p[0] < 0.5:
        if ret == -1:
            ret = 1
    else:
        if ret == -1:
            ret = 0
    return 1 - ret

n = 512
x = ti.field(dtype=ti.f32, shape=(n, n))


@ti.kernel
def paint():
    for i, j in ti.ndrange(n * 4, n * 4):
        # 4x4 super sampling:
        ret = taichi_logo(ti.Vector([i, j]) / (n * 4))
        x[i // 4, j // 4] += ret / 16


def main():
    paint()

    gui = ti.GUI('Logo', (n, n))
    while gui.running:
        gui.set_image(x)
        gui.show()


if __name__ == '__main__':
    main()
