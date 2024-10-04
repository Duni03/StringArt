import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.draw import line, circle_perimeter, line_aa, ellipse_perimeter
from skimage.feature import canny
import copy
from math import atan2
from skimage.transform import resize
from image_transformations import *
from time import time

def plot_grayscale_pic(picture, fig_width=7, fig_length=7):
    plt.figure(figsize=(fig_width,fig_length))
    plt.imshow(picture, cmap=plt.get_cmap("gray"), vmin=0.0, vmax=1.0)
    plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

LONG_SIDE = 300

img = mpimg.imread("clrtest.jpg")
img = largest_square(img)
img = resize(img, (LONG_SIDE, LONG_SIDE))

orig_pic_clr = copy.deepcopy(img)

orig_pic = rgb2gray(img)

edges = canny(orig_pic, sigma=2)
# orig_pic[edges] = 0.3
orig_pic = orig_pic*0.9

plot_grayscale_pic(orig_pic)
plot_grayscale_pic(edges)

orig_pic[0][:5]

def create_rectangle_nail_positions(picture, nail_step=2):
    height = len(picture)
    width = len(picture[0])

    nails_top = [(0, i) for i in range(0, width, nail_step)]
    nails_bot = [(height-1, i) for i in range(0, width, nail_step)]
    nails_right = [(i, width-1) for i in range(1, height-1, nail_step)]
    nails_left = [(i, 0) for i in range(1, height-1, nail_step)]
    nails = nails_top + nails_right + nails_bot + nails_left
    print(len(nails),nails)
    return nails

def create_circle_nail_positions(picture, nail_step=2):
    height = len(picture)
    width = len(picture[0])

    centre = (height // 2, width // 2)
    radius = min(height, width) // 2 - 1
    rr, cc = circle_perimeter(centre[0], centre[1], radius)
    nails = list(set([(rr[i], cc[i]) for i in range(len(cc))]))
    nails.sort(key=lambda c: atan2(c[0] - centre[0], c[1] - centre[1]))
    print(len(nails),nails)
    nails = nails[::nail_step]

    return nails

def init_black_canvas(picture):
    height = len(picture)
    width = len(picture[0])
    return np.zeros((height, width))

def init_white_canvas(picture):
    height = len(picture)
    width = len(picture[0])
    return np.ones((height, width))

def get_aa_line(from_pos, to_pos, str_strength, picture):
    rr, cc, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    line = picture[rr, cc] + str_strength * val
    line = np.clip(line, a_min=0, a_max=1)

    return line, rr, cc

def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength):

    best_cumulative_improvement = -99999
    best_nail_position = None
    best_nail_idx = None

    for nail_idx, nail_position in enumerate(nails):

        overlayed_line, rr, cc = get_aa_line(current_position, nail_position, str_strength, str_pic)

        before_overlayed_line_diff = np.abs(str_pic[rr, cc] - orig_pic[rr, cc])**2
        after_overlayed_line_diff = np.abs(overlayed_line - orig_pic[rr, cc])**2

        cumulative_improvement =  np.sum(before_overlayed_line_diff - after_overlayed_line_diff)

        if cumulative_improvement >= best_cumulative_improvement:
            best_cumulative_improvement = cumulative_improvement
            best_nail_position = nail_position
            best_nail_idx = nail_idx

    return best_nail_idx, best_nail_position, best_cumulative_improvement

def create_art(nails, orig_pic, str_pic, str_strength):

    start = time()
    iter_times = []

    current_position = nails[0]
    pull_order = [0]

    i = 0
    while True:
        start_iter = time()

        i += 1
        if i > 2000:
            break
        if i % 200 == 0:
            plot_grayscale_pic(str_pic)

        idx, best_nail_position, best_cumulative_improvement = find_best_nail_position(current_position, nails,
                                                                                  str_pic, orig_pic, str_strength)
        pull_order.append(idx)
        best_overlayed_line, rr, cc = get_aa_line(current_position, best_nail_position, str_strength, str_pic)
        str_pic[rr, cc] = best_overlayed_line

        current_position = best_nail_position
        iter_times.append(time() - start_iter)

    print(f"Time: {time() - start}")
    print(f"Avg iteration time: {np.mean(iter_times)}")
    return pull_order

nails = create_circle_nail_positions(orig_pic, 4)
# nails = create_rectangle_nail_positions(orig_pic, 4)
str_pic = init_white_canvas(orig_pic)
pull_order = create_art(nails, orig_pic, str_pic, -0.05)

for i in nails:
    plt.plot(i[0], i[1],"x")
plt.show()

print(f"Thread pull order by nail index:\n{'-'.join([str(idx) for idx in pull_order])}")

nails = create_circle_nail_positions(orig_pic, 4)
str_pic = init_black_canvas(orig_pic)
create_art(nails, orig_pic, str_pic, 0.05)

def scale_nails(x_ratio, y_ratio, nails):
    return [(int(y_ratio*nail[0]), int(x_ratio*nail[1])) for nail in nails]

def pull_order_to_array_bw(order, canvas, nails):
    # Draw a black and white pull order on the defined resolution

    for pull_start, pull_end in zip(order, order[1:]):  # pairwise iteration
        rr, cc, val = line_aa(nails[pull_start][0], nails[pull_start][1],
                              nails[pull_end][0], nails[pull_end][1])
        canvas[rr, cc] += val * -0.1

    return np.clip(canvas, a_min=0, a_max=1)

def pull_order_to_array_rgb(orders, canvas, nails, colors):
    color_order_iterators = [iter(zip(order, order[1:])) for order in orders]
    for _ in range(len(orders[0]) - 1):
        # pull colors alternately
        for color_idx, iterator in enumerate(color_order_iterators):
            pull_start, pull_end = next(iterator)
            rr_aa, cc_aa, val_aa = line_aa(
                nails[pull_start][0], nails[pull_start][1],
                nails[pull_end][0], nails[pull_end][1]
            )

            val_aa_colored = np.zeros((val_aa.shape[0], len(colors)))
            for idx in range(len(val_aa)):
                val_aa_colored[idx] = np.full(len(colors), val_aa[idx])

            canvas[rr_aa, cc_aa] += colors[color_idx] * val_aa_colored * -0.1

            # rr, cc = line(
            #     nails[pull_start][0], nails[pull_start][1],
            #     nails[pull_end][0], nails[pull_end][1]
            # )
            # canvas[rr, cc] = colors[color_idx]
    return np.clip(canvas, a_min=0, a_max=1)

image_dimens = 900, 900
blank = np.ones(image_dimens)
scaled_nails = scale_nails(
    image_dimens[1] / len(orig_pic),
    image_dimens[0] / len(orig_pic[0]),
    nails
)

result = pull_order_to_array_bw(
    pull_order,
    blank,
    scaled_nails
)
mpimg.imsave('output.png', result, cmap=plt.get_cmap("gray"), vmin=0.0, vmax=1.0)

pull_orders = []
nails = create_circle_nail_positions(orig_pic_clr, 4)

r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]

orig_pic = r
str_pic_r = init_white_canvas(orig_pic)
pull_orders.append(create_art(nails, orig_pic, str_pic_r, -0.1))

orig_pic = g
str_pic_g = init_white_canvas(orig_pic)
pull_orders.append(create_art(nails, orig_pic, str_pic_g, -0.1))

orig_pic = b
str_pic_b = init_white_canvas(orig_pic)
pull_orders.append(create_art(nails, orig_pic, str_pic_b, -0.1))

pic = [list(zip(str_pic_r[idx], str_pic_g[idx], str_pic_b[idx])) for idx in range(len(r))]

print(f"Thread pull order by nail index:\n{'-'.join([str(idx) for idx in pull_orders])}")
len(pull_orders[0])

color_image_dimens = 900, 900, 3
blank = np.ones(color_image_dimens)
scaled_nails = scale_nails(
    color_image_dimens[1] / len(orig_pic),
    color_image_dimens[0] / len(orig_pic[0]),
    nails
)

result = pull_order_to_array_rgb(
    pull_orders,
    blank,
    scaled_nails,
    (np.array((1., 0., 0.,)), np.array((0., 1., 0.,)), np.array((0., 0., 1.,)))
)
mpimg.imsave('output-color.png', result, vmin=0.0, vmax=1.0)
