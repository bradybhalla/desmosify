# USAGE: python3 to_desmos.py <image path> [<clusters> <color weight> <blur level>]
# <image path>: str, the path to the image
# <clusters>: int, number of clusters
# <color weight>: float, how much clusering depends on color vs. location
# <blur level>: int, how much blur should be applied to the image

import sys
import numpy as np
import matplotlib.pyplot as plt

from order_outlines import order_outlines
from simplify_image import simplify_image
from fourier_series import Fourier_Series


def get_js(desmos_code, color):
	replaced = desmos_code.replace("\\","\\\\")
	color_hex = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
	return f"{{latex:\"{replaced}\", color:\"{color_hex}\", parametricDomain:{{min:\"0\", max:\"a\"}}, fill:true, lines:true, fillOpacity:1}}"

if __name__ == "__main__":
	# minimum points in outline for Fourier Series conversion
	MIN_POINTS = 30

	js_codes = ["{latex:\"a=0\",sliderBounds:{min:\"0\", max:\"2\\\\pi\"}}", "{latex:\"N=[0...25]\"}"]

	unordered_js_shape_codes = []

	# break image into sections
	try:
		image_path = sys.argv[1]

		if len(sys.argv) >= 3:
			clusters = int(sys.argv[2])
		else:
			clusters = 70

		if len(sys.argv) >= 4:
			color_weight = float(sys.argv[3])
		else:
			color_weight = 1
		
		if len(sys.argv) >= 5:
			blur_level = int(sys.argv[4])
		else:
			blur_level = 2
		
	except:
		print("Invalid arguments")
		sys.exit()
	sections, outlines = simplify_image(
		image_path,
		COLOR_WEIGHT=color_weight,
		CLUSTERS=clusters,
		BLUR_LEVEL=2*blur_level+1,
		show_steps=True # set to False to avoid popup windows
	)

	next_print = len(sections)/10
	for i in range(len(sections)):
		if i > next_print:
			print(f"{i}/{len(sections)} sections completed")
			next_print += len(sections)/10

		color = sections[i][-3:]

		# if it is white, skip
		#if (color > 240).all():
		#	continue

		# get all outlines in the section and order them into paths
		if len(outlines[i]) < MIN_POINTS:
			continue
		all_ordered_outlines = order_outlines(outlines[i])

		for ordered_outline in all_ordered_outlines:
			# if there aren't enough points, skip
			if len(ordered_outline) < MIN_POINTS:
				continue

			# calculate and plot fourier series
			points = np.array(ordered_outline)
			X, Y = points.T[1], -points.T[0]

			FS = Fourier_Series(X,Y)

			FS.plot(25,color=color/255)

			# convert fourier series into JS for desmos
			desmos_code = FS.desmosify(25)

			# area calculated from first term of FS and Green's theorem!!
			approx_area = abs(np.math.pi*(FS.x_coeffs[1]["cos"]*FS.y_coeffs[1]["sin"] - FS.x_coeffs[1]["sin"]*FS.y_coeffs[1]["cos"]))
			unordered_js_shape_codes.append((-approx_area, get_js(desmos_code, color)))

	# sort shapes in order of descending size and add to js_codes list
	unordered_js_shape_codes.sort()
	for size, code in unordered_js_shape_codes:
		js_codes.append(code)

	plt.title("Fourier Series drawing")
	plt.show()

	# write JS for desmos into "out.txt"
	with open("out.txt", "w") as f:
		js_arr = ",".join(js_codes)
		f.write(f"Calc.setExpressions([{js_arr}]);")

