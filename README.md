# desmosify
Convert an image into a [Desmos](https://desmos.com/calculator) graph!

<img src="images/example.png" alt="Before" width="200"/>
<img src="images/result.png" alt="After" width="200"/>

### Usage
For a conversion with preset values, use `python3 desmosify.py <image path>`. For more information, run `python3 desmosify.py --help`.

The this program will print JavaScript to stdout. Copy and paste this output into the JavaScript console in [Desmos](https://desmos.com/calculator) and zoom out to see the image. Note that the output is very large so it might be a good idea to save to a file before copying (or pipe to `pbcopy`).

### Requirements
These are the versions which worked for me. It is likely but not guaranteed that other versions of Python/packages will also work.
- Python 3.10.14
- numpy==1.26.4
- matplotlib==3.8.4
- scipy==1.13.0
- opencv-python==4.10.0.82
- scikit-learn==1.5.0

