import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

from utils.fourier_series import (
    points_to_fs,
    fs_to_desmos,
    terms_needed_for_err
)

from utils.cluster import cluster_image, combine_similar_clusters, split_region


def show(image, title=""):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert an image to Desmos.")
    parser.add_argument("image_path", type=str, help="Path to the image to convert.")
    parser.add_argument("--clusters", type=int, default=200, help="Number of clusters.")
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=0.2,
        help="Weight of position in clustering (which uses combined position and color data).",
    )
    parser.add_argument(
        "--min-color-dist",
        type=float,
        default=20 / 255,
        help="Minimum L2 distance of RGB colors before clusters are combined.",
    )
    parser.add_argument(
        "--area-thresh",
        type=float,
        default=20,
        help="Smallest area of region for which a Fourier Series is created. Other regions are filled with points.",
    )
    parser.add_argument(
        "--err-thresh",
        type=float,
        default=0.2,
        help="Maximum allowable mean L2 distance between fourier series points and original points.",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=60,
        help="Maximum number of terms in a Fourier Series.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print progress messages to stderr",
    )

    args = parser.parse_args()

    # load image
    im = cv.imread(args.image_path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32) / 255
    height, width, _ = im.shape

    new_width = 512
    new_height = 512
    im = cv.resize(im, (new_width, new_height))

    # find initial clustering
    clustered_im, colors, _ = cluster_image(
        im, args.clusters, position_weight=args.pos_weight
    )

    if args.verbose:
        print(f"Done clustering ({len(colors)} clusters)", file=sys.stderr)

    # combine clusters with similar color
    clustered_im, colors = combine_similar_clusters(
        clustered_im, colors, max_dist_to_combine=args.min_color_dist
    )
    if args.verbose:
        print(f"Done combining clusters ({len(colors)} unique colors)", file=sys.stderr)

    all_fourier_series_eqns = []
    all_point_eqns = []
    for i, c in enumerate(colors):
        r, g, b = (c * 255).astype(np.uint8)
        color_str = f"rgb({r},{g},{b})"

        mask = cv.flip((clustered_im == i).astype(np.uint8), 0)

        large_region_mask, small_region_mask = split_region(
            mask, area_threshold=args.area_thresh
        )

        # process large regions into fourier series
        contours, _ = cv.findContours(
            large_region_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
        )
        for pixels in contours:
            area = cv.contourArea(pixels)

            if pixels.shape[0] <= 2:
                raise ValueError("this region is too small")

            X = pixels[:, 0, 0]
            Y = pixels[:, 0, 1]

            fs = points_to_fs(X, Y)

            terms = terms_needed_for_err(
                fs, X, Y, mean_err_threshold=args.err_thresh, max_terms=args.max_terms
            )
            equation_object = {
                "latex": fs_to_desmos(*fs, max_terms=terms),
                "color": color_str,
                "lines": True,
                "fill": True,
                "fillOpacity": 1,
                "lineWidth": 4,
            }
            all_fourier_series_eqns.append((area, json.dumps(equation_object)))

        # process small regions into points
        points_eqn = {
            "latex": f"[{','.join([f'({b},{a})' for a,b in zip(*np.where(small_region_mask==1))])}]",
            "color": color_str,
            "pointSize": 2,
            "pointOpacity": 1,
        }
        all_point_eqns.append(json.dumps(points_eqn))

    all_fourier_series_eqns.sort(key=lambda x: x[0], reverse=True)

    all_equations = [json.dumps({"latex": rf"N=\left[0...{args.max_terms}\right]"})]
    all_equations += [i[1] for i in all_fourier_series_eqns]
    all_equations += all_point_eqns
    print(f"Calc.setExpressions([{','.join(all_equations)}])")

    if args.verbose:
        print(
            f"Done generating equations ({len(all_equations)} equations)",
            file=sys.stderr,
        )
