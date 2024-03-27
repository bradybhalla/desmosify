# cluster an image into sections which account for color and location
# turn these sections into outlines

import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import KMeans


def show(image, title=""):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()

def cluster_image(im, n_clusters, position_weight=0):
    index_arr = np.indices((im.shape[0],im.shape[1])).reshape((2,im.shape[0]*im.shape[1])).T 
    color_arr = im[*index_arr.T]
    normalized_indices = index_arr / np.array([im.shape[0], im.shape[1]])
    normalized_colors = color_arr / 255 # pyright: ignore

    X = np.hstack([normalized_indices*position_weight, normalized_colors])
    model = KMeans(n_clusters=n_clusters, n_init="auto")
    model.fit(X)
    colors = model.cluster_centers_[:,-3:]
    preds = model.predict(X)

    clustered_img = np.zeros((im.shape[0], im.shape[1]), dtype=int)
    clustered_img[*index_arr.T] = preds

    return clustered_img, colors

def get_toplevel_polygon_equation(contours):
    all_polygons = []
    for i in contours:
        points = i[:,0,:]
        if len(points) >= 3:
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

    im = cv.line(im, outline_point, hole_point, 0, 1)

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
    to_rgb = lambda x: np.array([int(i + j, 16) for i, j in zip(x[1::2], x[2::2])]) / 255
    graph_colors = {
            "r" : to_rgb("#c74440"),
            "b" : to_rgb("#2d70b3"),
            "g" : to_rgb("#388c46"),
            "p" : to_rgb("#6042a6"),
            "o" : to_rgb("#fa7e19"),
            "k" : to_rgb("#000000"),
            "W" : to_rgb("#ffffff"),
            "c" : to_rgb("#00ffff"),
            }
    graphs = "bgpkk"

    possible_colors = []
    for i in range(2**len(graphs)):
        format_str = f"{{:0{len(graphs)}b}}"
        used = [j=="1" for j in format_str.format(i)]
        color = np.ones(3)
        for k,j in enumerate(used):
            if j:
                color = 0.6*color + 0.4*graph_colors[graphs[k]]
        possible_colors.append(color)
    possible_colors = np.array(possible_colors)


    # load image
    im = cv.imread("images/hollow.jpg")
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    height, width, _ = im.shape

    new_width = 300
    im = cv.resize(im, (new_width, new_width * height // width))

    clusters, colors = cluster_image(im, 200, position_weight=0)

    im = colors[clusters]
    # show(im)

    YUV_of_RGB = np.array([[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]])
    color_dists = lambda c: np.linalg.norm((possible_colors - c) @ YUV_of_RGB.T, axis=1)

    # new_color_indices contains the index into possible_colors of each of the clusters
    new_color_indices = np.array([np.argmin(color_dists(i)) for i in colors])
    im = possible_colors[new_color_indices][clusters]
    show(im, "target")

    # ind_to_colors maps possible_colors elements to the colors which make them up
    ind_to_colors = {}
    for i in set(new_color_indices):
        format_str = f"{{:0{len(graphs)}b}}"
        used = [j=="1" for j in format_str.format(i)]
        ind_to_colors[i] = [j for j,k in zip(graphs, used) if k]

    # graph contents contains the color of each graph and
    # which clusters it needs to show
    graph_contents = []
    for c in graphs:
        contents = []
        for i,j in ind_to_colors.items():
            if len(j) == 0:
                continue
            if j[0] == c:
                contents += list(np.where(new_color_indices == i)[0])
                del j[0]
        graph_contents.append((c, contents))

    print("Colors calculated", file=sys.stderr)

    plots = []
    total_im = np.ones(im.shape)
    for i, (c,clusters_needed) in enumerate(graph_contents):
        im = np.zeros(total_im.shape[:2])
        for cl in clusters_needed:
            im += clusters == cl
        im = im.astype(np.uint8)
        # show(im, title=c)

        print(f"Computing equation {i+1}/{len(graph_contents)}", file=sys.stderr)
        im, eq = binary_to_equation(im)
        plots.append(eq)

        total_im[np.where(im)] = 0.6*total_im[np.where(im)] + 0.4*graph_colors[c]

    print("\n".join(plots), end="")
    show(total_im, "reconstructed")

