# cluster an image into sections which account for color and location
# turn these sections into outlines

import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def show(image, title=""):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()

def get_toplevel_polygon_equation(contours):
    all_polygons = []
    for i in contours:
        points = i[:,0,:]
        if len(points) >= 3:
            # coeffs = points_to_fs(points[:,0], points[:,1])
            # all_polygons.append(fs_to_desmos(*coeffs))
            polygon = r"\operatorname{polygon}\left("
            polygon += ",".join([rf"\left({j[0]}, {im.shape[0]-j[1]}\right)" for j in points])
            polygon += r"\right)"
            all_polygons.append(polygon)
    return r"\left[" + ",".join(all_polygons) + r"\right]"

def remove_hole(im, contours, hierarchy, contour_index):
    contour_points = contours[contour_index][:,0,:]

    holes_of_contour = np.argwhere(hierarchy[0,:,3] == contour_index)[:,0]

    outline_point = None
    hole_point = None
    min_dist = float("inf")
    for i in holes_of_contour:
        hole_points = contours[i][:,0,:]
        for hp in hole_points:
            closest_op_ind = np.argmin(np.linalg.norm(contour_points - hp, axis=1))
            closest_op = contour_points[closest_op_ind]
            dist = np.linalg.norm(closest_op - hp)
            if dist < min_dist:
                outline_point = closest_op
                hole_point = hp
                min_dist = dist

    im = cv.line(im, outline_point, hole_point, 0, 1) #pyright: ignore

    return im

def binary_to_equation(im):
    kernel = np.ones((3, 3), np.uint8)
    kernel[0,0] = 0
    kernel[0,2] = 0
    kernel[2,0] = 0
    kernel[2,2] = 0
    im = cv.dilate(im, kernel)
    im = cv.erode(im, kernel)
    im = cv.erode(im, kernel)
    im = cv.dilate(im, kernel)

    while True:
        # RETR_CCOMP sets hierarchy to filled contours and holes
        contours, hierarchy = cv.findContours(im, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
        if hierarchy is None or np.max(hierarchy[0,:,3]) == -1:
            # no holes left
            break

        contours_with_holes = np.argwhere(hierarchy[0,:,2] != -1)[:,0]
        for i in contours_with_holes:
            im = remove_hole(im, contours, hierarchy, i)

    # only gives top level
    contours, _ = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
    return im, get_toplevel_polygon_equation(contours)


if __name__ == "__main__":
    to_rgb = lambda x: np.array([int(i + j, 16) for i, j in zip(x[1::2], x[2::2])])
    graph_colors = {
            "r" : (to_rgb("#c74440"), 0.4),
            "b" : (to_rgb("#2d70b3"), 0.4),
            "g" : (to_rgb("#388c46"), 0.4),
            "p" : (to_rgb("#6042a6"), 0.4),
            "o" : (to_rgb("#fa7e19"), 0.4),
            "k" : (to_rgb("#000000"), 0.4),
            "K" : (to_rgb("#000000"), 1),
            "B" : (to_rgb("#2d70b3"), 1),
            "W" : (to_rgb("#ffffff"), 0.4),
            "c" : (to_rgb("#00ffff"), 0.4),
            }
    graphs = "Krb"

    possible_colors = np.zeros((2**len(graphs), 3)) + 255
    color_inds = np.arange(2**len(graphs))
    for i,j in enumerate(graphs):
        colors_using_graph = np.where(np.bitwise_and(color_inds, 2**i) != 0)
        c, a = graph_colors[j]
        possible_colors[colors_using_graph] *= 1 - a
        possible_colors[colors_using_graph] += a * c
    possible_colors = possible_colors.astype(int)


    # load image
    im = cv.imread("images/hollow.jpg")
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    height, width, _ = im.shape

    new_width = 800
    im = cv.resize(im, (new_width, new_width * height // width))
    # show(im, "original")


    best_colors = np.expand_dims(im, axis=2)
    best_colors = np.repeat(best_colors, possible_colors.shape[0], axis=2)
    best_colors = best_colors - possible_colors
    best_colors = np.linalg.norm(best_colors, axis=3)
    best_colors = np.argmin(best_colors, axis=2)

    im = possible_colors[best_colors]
    show(im, "desmos color target")

    print("Colors calculated", file=sys.stderr)

    plots = []
    total_im = np.ones(im.shape)
    for i, j in enumerate(graphs):
        im = np.zeros(total_im.shape[:2])
        im = np.bitwise_and(best_colors, 2**i) // 2**i
        im = im.astype(np.uint8)
        # show(im, title=c)

        print(f"Computing equation {i+1}/{len(graphs)}", file=sys.stderr)
        im, eq = binary_to_equation(im)
        plots.append(eq)

        c, a = graph_colors[j]

        total_im[np.where(im)] = (1-a)*total_im[np.where(im)] + a*c/255

    print("\n".join(plots), end="")
    show(total_im, "reconstructed")
