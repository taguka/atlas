import os
import cv2
import numpy as np


BASEPATH = '/data/amazon'
PNGPATH = os.path.join(BASEPATH, 'test')



def find_inputs(folder, types=('.png'), prefix=''):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if prefix and base.startswith(prefix) and ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((base, abs_filename))
    return inputs


def main():
    png_inputs = find_inputs(PNGPATH, types=('.png',))
    png_stats = []
    flags = cv2.IMREAD_GRAYSCALE
    for f in png_inputs:
        img = cv2.imread(f[1],flags)
        mean, std = cv2.meanStdDev(img)
        png_stats.append(np.array([mean[::-1] / 255, std[::-1] / 255]))
    png_vals = np.mean(png_stats, axis=0)
    print(png_vals)


if __name__ == '__main__':
    main()

