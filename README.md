# desmosify
Convert an image into a [Desmos](https://desmos.com/calculator) graph!

<img src="images/example.png" alt="Before" width="500"/>
<img src="images/result.png" alt="After" width="500"/>

## Usage
For a conversion with preset values, use `python3 desmosify.py <image path>`. For more information, run `python3 desmosify.py --help`.

The this program will print JavaScript to stdout. Copy and paste this output into the JavaScript console in [Desmos](https://desmos.com/calculator) and zoom out to see the image. Note that the output is very large so it might be a good idea to save to a file before copying (or pipe to `pbcopy`).

## How it works
The first step is clustering the image into regions of constant color, which is done using k-means clustering. The color and pixel locations are combined into a 5-dimensional space so the clusters group regions of pixels with similar location and color. Clusters with similar color centers are combined, which completes the partitioning of pixels into regions of constant color.

Each region is then split into connected components. Components with larger areas are turned into equations by calculating a Fourier series approximation of the outline. This is plotted as a filled parametric curve in Desmos. Components with smaller areas are simply plotted as a set of points.

The program outputs the JavaScript code which generates the equations and points using the Desmos API, so it can be pasted into the web console in order to show the final graph.

## Requirements
These are the versions which worked for me. It is likely but not guaranteed that other versions of Python/packages will also work.
- Python 3.10.14
- numpy==1.26.4
- matplotlib==3.8.4
- scipy==1.13.0
- opencv-python==4.10.0.82
- scikit-learn==1.5.0

