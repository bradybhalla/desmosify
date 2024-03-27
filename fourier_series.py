import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def get_interpolator(X, Y):
    """
    Create a function f which interpolates the given points.
    f(0) is the start and f(1) is the end, with the in between
    values corresponding to the distance traveled.

    Arguments:
        - X: a 1D numpy array
        - Y: a 1D numpy array

    Returns:
        - f: [0,1] -> R^2
    """
    # filter out consecutive equal values
    inds = np.where(np.logical_or(X[1:] != X[:-1], Y[1:] != Y[:-1]))
    X = np.concatenate([X[inds], [X[-1]]])
    Y = np.concatenate([Y[inds], [Y[-1]]])

    # calculate time based on distance between consecutive points
    x_diffs = np.ediff1d(X, to_begin=0)
    y_diffs = np.ediff1d(Y, to_begin=0)
    diffs = np.sqrt(x_diffs**2 + y_diffs**2)
    t = (np.cumsum(diffs) / np.sum(diffs))[None, :]

    return RegularGridInterpolator(t, np.array([X, Y]).T)


def fourier_series_1d(x):
    """
    Calculate the Fourier series coefficients from the data. If N is
    the length of the input, the Fourier series approximates
    ([0, 1/N, ..., 1], x). Only works for real inputs.

    Arguments:
        - x: a 1D numpy array

    Returns:
        - a: coefficients on the term a[i]cos(2pi i t)
        - b: coefficients on the term b[i]sin(2pi i t)
    """
    N = x.shape[0]
    f = np.fft.fft(x) / N

    coeffs = 2*f[:(N+1)//2]
    coeffs[0] /= 2
    return np.real(coeffs), np.imag(coeffs)


def points_to_fs(X, Y):
    """
    Get Fourier series coefficients for x and y components of
    the interpolated data.

    Arguments:
        - X: a 1D numpy array
        - Y: a 1D numpy array

    Returns:
        - the coefficients of the two Fourier series
    """
    f = get_interpolator(X, Y)
    fx = f(np.linspace(0, 1, 2048))
    ax, bx = fourier_series_1d(fx[:, 0])
    ay, by = fourier_series_1d(fx[:, 1])
    return ax, bx, ay, by


def series_to_desmos(ax, bx, ay, by, max_terms=30):
    """
    Create a Desmos equation which traces a Fourier series approximation
    of the given points. In order for this to work, also define
    `N=[0...max_terms]`.

    Arguments:
        - ax, bx: fourier series coeffs of X coordinate
        - ay, by: fourier series coeffs of Y coordinate
        - max_terms: the number of terms to put in the equation

    Returns:
        - a string which can be pasted into Desmos
    """
    As_x = [rf"{ax[i]:.6f}" for i in range(max_terms)]
    Bs_x = [rf"{bx[i]:.6f}" for i in range(max_terms)]
    As_y = [rf"{ay[i]:.6f}" for i in range(max_terms)]
    Bs_y = [rf"{by[i]:.6f}" for i in range(max_terms)]

    return rf"\left(\operatorname{{total}}\left(\left[{', '.join(As_x)}\right]\cos\left(2\pi Nt\right)\right)+\operatorname{{total}}\left(\left[{', '.join(Bs_x)}\right]\sin\left(2\pi Nt\right)\right),\operatorname{{total}}\left(\left[{', '.join(As_y)}\right]\cos\left(2\pi Nt\right)\right)+\operatorname{{total}}\left(\left[{', '.join(Bs_y)}\right]\sin\left(2\pi Nt\right)\right)\right)"


def series_to_func(ax, bx, ay, by, terms=30):
    """
    Create a function which traces a Fourier series approximation
    of the given points.

    Arguments:
        - ax, bx: fourier series coeffs of X coordinate
        - ay, by: fourier series coeffs of Y coordinate
        - terms: the number of terms to use in the calculation

    Returns:
        - a function f(t) which traces out the points as t ranges
          from 0 to 1
    """
    ax, bx = ax[:terms, None], bx[:terms, None]
    ay, by = ay[:terms, None], by[:terms, None]
    i = np.arange(terms)[:, None]

    def f(t):
        theta = 2 * np.pi * (i * t).T
        return np.hstack(
            [
                np.cos(theta) @ ax + np.sin(theta) @ bx,
                np.cos(theta) @ ay + np.sin(theta) @ by,
            ]
        )

    return f


# example
if __name__ == "__main__":
    # X,Y are points of shape (it is a nice shape)
    X = np.array( [ 204, 192, 174, 162, 154, 144, 120, 101, 104, 109, 110, 112, 113, 95, 74, 74, 75, 75, 114, 115, 114, 113, 114, 114, 132, 157, 170, 226, 268, 311, 333, 342, 345, 346, 346, 312, 305, 360, 389, 358, 334, 309, 301, 301, 312, 328, 338, 342, 343, 343, 343, 342, 336, 336, 332, 327, 319, 304, 280, 275, 273, 257, 235, 218, 203, 202])

    Y = np.array( [ 330, 327, 337, 368, 404, 425, 429, 405, 354, 321, 283, 245, 232, 230, 227, 227, 169, 157, 155, 189, 233, 205, 162, 152, 97, 70, 63, 46, 44, 57, 77, 98, 115, 127, 127, 133, 173, 187, 148, 131, 127, 129, 148, 166, 180, 183, 185, 186, 186, 204, 235, 248, 311, 319, 368, 398, 420, 437, 423, 396, 372, 349, 336, 331, 330, 329])

    Y = -np.array(Y)

    ax, bx, ay, by = points_to_fs(X, Y)

    # plot shape
    plt.plot(X, Y, "o")

    # print desmos
    print(series_to_desmos(ax, bx, ay, by), end="")

    # get Fourier series function
    fs = series_to_func(ax, bx, ay, by)
    t = np.linspace(0,1,300)

    plt.plot(*fs(t).T)
    plt.show()
