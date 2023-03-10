# take a path and approximate its Fourier Series

import numpy as np
import matplotlib.pyplot as plt

# constants
PI = np.pi
TWO_PI = 2*np.pi
E = np.e

# assume points are processed to be in correct order already


class Fourier_Series:
    def __init__(self, X, Y, max_terms=100):
        # complex points of shape
        self.Z = np.array(X) + 1j*np.array(Y)

        # number of terms
        self.max_terms = max_terms

        # time points
        self.T = np.linspace(0, TWO_PI, len(self.Z))

        # exponential coefficients
        self.exp_coeffs = {}
        for i in range(-self.max_terms, self.max_terms+1):
            self.exp_coeffs[i] = self.get_exp_coeff(i)

        # real coefficients
        # ...[n] = {"cos":-0.5, "sin":0.2}
        self.x_coeffs = {}
        self.y_coeffs = {}
        self._gen_real_coeffs()

    # get coeff of term in exponential series

    def get_exp_coeff(self, n):
        return self._approx_int(self.T, self._fs_func(n))/TWO_PI

    # integral of a TWO_PI periodic function
    # with points (T,F)
    def _approx_int(self, T, F):
        total = 0
        for i in range(1, len(T)):
            dt = (T[i]-T[i-1]) % TWO_PI
            f = (F[i] + F[i-1])/2
            total += f*dt
        dt = (T[0] - T[len(T)-1]) % TWO_PI
        f = (F[0] + F[len(T)-1])/2
        total += f*dt

        return total

    # returns the points of f(t)e^{-int}
    def _fs_func(self, n):
        return self.Z*E**(-1j*self.T*n)

    # generate real coefficients
    def _gen_real_coeffs(self):
        self.x_coeffs[0] = {
            "cos": self.exp_coeffs[0].real,
            "sin": 0
        }
        self.y_coeffs[0] = {
            "cos": self.exp_coeffs[0].imag,
            "sin": 0
        }

        for i in range(1, self.max_terms+1):
            self.x_coeffs[i] = {
                "cos": self.exp_coeffs[i].real + self.exp_coeffs[-i].real,
                "sin": self.exp_coeffs[-i].imag - self.exp_coeffs[i].imag
            }
            self.y_coeffs[i] = {
                "cos": self.exp_coeffs[i].imag + self.exp_coeffs[-i].imag,
                "sin": self.exp_coeffs[i].real - self.exp_coeffs[-i].real
            }

    # evaluate fourier series at times T
    def evaluate(self, T, terms):
        Z = np.zeros(len(T))

        for i in range(-terms, terms+1):
            Z = Z + self.exp_coeffs[i]*E**(1j*i*T)

        return Z

    # plot fourier series
    def plot(self, terms, *args, **kwargs):
        T = np.linspace(0, TWO_PI, 100)
        Z = self.evaluate(T, terms)

        plt.plot(Z.real, Z.imag, *args, **kwargs)

    # generate desmos code to draw fourier series
    # also must add the line N=[0...terms]
    def desmosify(self, terms, round_to=4):
        x_cos_coeffs = [str(round(self.x_coeffs[i]["cos"], round_to))
                        for i in range(terms+1)]
        x_sin_coeffs = [str(round(self.x_coeffs[i]["sin"], round_to))
                        for i in range(terms+1)]
        y_cos_coeffs = [str(round(self.y_coeffs[i]["cos"], round_to))
                        for i in range(terms+1)]
        y_sin_coeffs = [str(round(self.y_coeffs[i]["sin"], round_to))
                        for i in range(terms+1)]

        S = "\\operatorname{{total}}\\left(\\left[{}\\right]\\left[N+1\\right]\\cdot\\cos\\left(Nt\\right) + \\left[{}\\right]\\left[N+1\\right]\\cdot\\sin\\left(Nt\\right)\\right)"
        both = (
            S.format(",".join(x_cos_coeffs), ",".join(x_sin_coeffs)),
            S.format(",".join(y_cos_coeffs), ",".join(y_sin_coeffs))
        )
        result = f"\\left({both[0]}, {both[1]}\\right)"
        return result


# example
if __name__ == "__main__":
    # X,Y are points of shape (it is a nice shape)
    X = np.array([204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 201, 198, 195, 192, 190, 189, 187, 186, 186, 185, 184, 183, 182, 180, 180, 179, 178, 177, 176, 174, 172, 169, 168, 167, 166, 165, 165, 164, 163, 163, 162, 162, 162, 162, 162, 162, 162, 161, 161, 161, 161, 161, 161, 161, 160, 160, 159, 158, 157, 156, 155, 154, 153, 152, 152, 151, 150, 150, 149, 148, 147, 146, 145, 144, 144, 144, 144, 144, 142, 140, 138, 137, 137, 137, 136, 135, 133, 132, 130, 128, 126, 124, 121, 120, 118, 116, 114, 113, 111, 109, 108, 107, 105, 104, 104, 103, 102, 102, 101, 101, 101, 101, 100, 100, 100, 100, 100, 100, 100, 101, 102, 102, 102, 103, 103, 104, 105, 105, 106, 107, 108, 108, 108, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110, 111, 111, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 111, 107, 104, 102, 100, 98, 95, 93, 90, 88, 87, 85, 84, 83, 83, 83, 83, 81, 79, 78, 76, 75, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 77, 79, 81, 84, 87, 90, 94, 97, 100, 104, 107, 110, 113, 114, 114, 114, 114, 114, 114, 114, 114, 115, 115, 115, 115, 115, 115, 115, 115, 115, 114, 113, 112, 112, 112, 112, 112, 112, 113, 113, 113, 113, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 115, 117, 118, 120, 121, 123, 124, 125, 126, 128, 129, 130, 132, 133, 135, 138, 140, 143, 145, 148, 150, 152, 153, 153, 154, 155, 156, 157, 157, 158, 161, 164, 168, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 170, 174, 179, 182, 186, 190, 195, 197, 199, 202, 204, 208, 212, 215, 219, 223, 226, 228, 231, 233, 236, 238, 241, 244, 247, 249, 251, 253, 256, 260, 263, 265, 268, 272, 276, 280, 283, 285, 287, 290, 293, 297, 299, 301, 303, 305, 307, 310, 311, 313, 315, 316, 318, 319, 320, 321, 323, 325, 327, 329, 330, 331, 332, 333, 333, 334, 335, 335, 336, 336, 337, 338, 338, 339, 339, 340, 341, 341, 341, 342, 342, 343, 343, 343, 343, 343, 344, 344, 344, 344, 344, 345, 345, 345, 345, 345, 345, 345, 345, 345, 345, 345, 345, 345, 345, 345, 345,
                 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 345, 343, 340, 337, 335, 334, 332, 329, 326, 323, 320, 317, 312, 308, 306, 303, 302, 301, 301, 300, 300, 301, 301, 302, 302, 303, 304, 304, 305, 307, 309, 311, 313, 316, 318, 320, 323, 326, 330, 334, 337, 342, 348, 354, 360, 364, 368, 373, 376, 379, 383, 385, 386, 388, 389, 390, 391, 391, 391, 390, 389, 388, 386, 384, 382, 380, 378, 375, 371, 368, 365, 362, 361, 360, 359, 359, 358, 358, 357, 356, 355, 354, 352, 351, 350, 348, 346, 345, 343, 340, 338, 335, 334, 333, 332, 330, 328, 325, 323, 322, 320, 318, 316, 314, 313, 312, 311, 310, 309, 308, 308, 308, 307, 306, 305, 304, 303, 303, 302, 302, 302, 302, 301, 301, 301, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 301, 301, 302, 302, 303, 303, 304, 305, 306, 306, 308, 308, 309, 310, 311, 311, 312, 313, 314, 315, 316, 316, 317, 318, 320, 322, 323, 325, 326, 327, 327, 327, 328, 328, 328, 329, 329, 330, 330, 331, 331, 332, 332, 332, 333, 334, 335, 336, 338, 339, 340, 340, 340, 340, 340, 340, 340, 340, 340, 341, 341, 342, 342, 342, 342, 342, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 343, 342, 342, 342, 341, 341, 339, 338, 337, 337, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 335, 335, 334, 334, 334, 333, 333, 333, 333, 333, 332, 332, 332, 332, 332, 331, 331, 331, 331, 330, 330, 329, 329, 329, 328, 328, 327, 326, 326, 325, 325, 324, 323, 323, 322, 322, 321, 321, 320, 320, 320, 319, 319, 318, 316, 315, 315, 314, 312, 311, 310, 310, 309, 309, 308, 307, 306, 305, 304, 303, 300, 297, 295, 294, 292, 291, 290, 289, 288, 287, 286, 284, 283, 281, 280, 279, 278, 278, 277, 277, 277, 276, 276, 276, 276, 276, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 274, 274, 274, 274, 273, 273, 273, 272, 271, 271, 270, 269, 269, 268, 266, 265, 264, 263, 262, 261, 259, 259, 257, 256, 255, 253, 251, 250, 248, 247, 246, 245, 244, 243, 242, 241, 240, 238, 235, 234, 232, 230, 229, 227, 226, 226, 225, 224, 223, 223, 222, 221, 220, 219, 218, 217, 216, 216, 215, 214, 213, 212, 212, 211, 211, 210, 209, 206, 205, 204, 203, 203, 203, 203, 203, 202, 202, 202, 202])
    Y = np.array([330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 329, 328, 327, 327, 327, 327, 327, 327, 327, 327, 327, 328, 329, 330, 331, 332, 333, 335, 337, 339, 342, 343, 345, 347, 349, 351, 353, 355, 357, 360, 362, 364, 366, 367, 368, 370, 372, 375, 379, 380, 382, 383, 384, 386, 389, 391, 393, 395, 398, 401, 404, 405, 407, 409, 411, 413, 415, 417, 419, 421, 423, 424, 425, 425, 425, 425, 425, 426, 427, 428, 428, 428, 428, 428, 428, 429, 429, 429, 429, 429, 429, 429, 429, 429, 428, 426, 425, 424, 422, 421, 419, 417, 416, 415, 413, 412, 410, 407, 405, 401, 399, 395, 391, 387, 385, 382, 378, 374, 370, 367, 365, 362, 359, 357, 354, 351, 349, 346, 343, 340, 336, 333, 330, 327, 324, 322, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 320, 315, 308, 302, 296, 291, 288, 283, 279, 275, 271, 266, 263, 260, 257, 254, 252, 252, 251, 251, 251, 250, 247, 245, 245, 245, 245, 244, 241, 241, 240, 240, 240, 240, 238, 235, 233, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 231, 231, 230, 230, 230, 230, 230, 229, 229, 229, 229, 229, 229, 229, 228, 228, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 226, 222, 218, 213, 208, 204, 199, 195, 192, 187, 182, 179, 176, 173, 169, 164, 162, 160, 159, 159, 159, 158, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 156, 156, 156, 156, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 158, 161, 165, 168, 169, 171, 173, 176, 180, 184, 189, 194, 198, 202, 206, 208, 211, 213, 214, 217, 220, 223, 226, 228, 230, 232, 233, 234, 234, 234, 234, 234, 233, 232, 228, 226, 223, 220, 217, 215, 211, 209, 205, 202, 199, 196, 193, 192, 189, 187, 185, 182, 179, 177, 174, 171, 167, 164, 162, 160, 158, 155, 153, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152, 149, 144, 136, 130, 125, 122, 118, 115, 112, 109, 105, 103, 100, 97, 94, 92, 89, 87, 84, 81, 79, 76, 75, 74, 73, 72, 72, 71, 70, 70, 69, 68, 66, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 63, 62, 60, 58, 57, 56, 54, 54, 53, 52, 51, 50, 49, 48, 47, 46, 46, 46, 45, 45, 45, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 46, 46, 47, 48, 48, 49, 50, 51, 52, 52, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 66, 68, 69, 71, 73, 74, 75, 76, 77, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 122, 123, 123, 125, 126, 126, 127, 127,
                 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 128, 130, 133, 137, 139, 144, 147, 151, 155, 157, 158, 161, 162, 164, 166, 168, 169, 171, 173, 175, 177, 178, 179, 181, 182, 183, 184, 185, 186, 187, 187, 187, 187, 187, 187, 187, 186, 184, 183, 180, 176, 174, 171, 167, 163, 160, 156, 154, 153, 150, 148, 146, 144, 143, 141, 139, 138, 136, 135, 134, 133, 132, 132, 132, 131, 131, 131, 131, 131, 131, 130, 130, 130, 130, 129, 129, 129, 128, 128, 128, 128, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 128, 128, 129, 129, 129, 130, 130, 131, 133, 134, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 148, 149, 150, 151, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 177, 178, 179, 179, 180, 180, 180, 181, 181, 181, 181, 181, 182, 182, 182, 182, 183, 183, 183, 183, 183, 183, 183, 183, 184, 184, 184, 184, 184, 184, 184, 184, 184, 185, 185, 185, 185, 185, 185, 185, 185, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 187, 189, 191, 193, 196, 199, 200, 202, 204, 206, 209, 211, 214, 216, 218, 220, 222, 224, 226, 228, 229, 232, 233, 234, 235, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 242, 248, 253, 258, 263, 268, 275, 282, 286, 292, 296, 299, 302, 304, 307, 309, 310, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 314, 319, 324, 330, 334, 336, 340, 344, 347, 351, 354, 356, 358, 360, 362, 365, 366, 368, 369, 372, 374, 377, 380, 382, 384, 386, 388, 390, 391, 392, 393, 394, 396, 398, 399, 400, 401, 403, 404, 406, 408, 410, 412, 413, 415, 416, 416, 418, 419, 420, 422, 425, 426, 427, 428, 430, 432, 433, 434, 434, 435, 435, 436, 436, 437, 437, 437, 437, 437, 436, 436, 436, 435, 434, 433, 433, 432, 430, 429, 426, 424, 423, 421, 419, 417, 415, 413, 412, 410, 409, 407, 405, 404, 403, 401, 400, 398, 396, 394, 393, 391, 389, 388, 386, 385, 384, 383, 381, 379, 377, 375, 374, 373, 372, 370, 369, 367, 366, 364, 363, 362, 360, 358, 357, 356, 355, 353, 352, 351, 349, 348, 347, 346, 344, 344, 343, 342, 341, 341, 340, 339, 339, 338, 337, 337, 336, 336, 335, 335, 334, 334, 334, 333, 333, 333, 333, 332, 332, 332, 332, 331, 331, 331, 331, 331, 331, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 329, 329, 329, 329, 329, 329, 329, 329])

    Y = -np.array(Y)

    # plot shape
    plt.plot(X, Y, "o")

    # get fourier series
    FS = Fourier_Series(X, Y)

    # plot fourier series
    FS.plot(25)

    # print desmos
    print(FS.desmosify(25))

    plt.show()
