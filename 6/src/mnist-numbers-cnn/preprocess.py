import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def preprocess_image(path):
    image = cv2.imread(path)

    # Grayscale image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    image = cv2.bitwise_not(image)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _ret, image = cv2.threshold(image, 128, 255, type=cv2.THRESH_BINARY)

    # Resizing before cropping into bounding box
    image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _ret, image = cv2.threshold(image, 128, 255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Cropping image
    while np.sum(image[0]) == 0:
        image = image[1:]
    while np.sum(image[:,0]) == 0:
        image = np.delete(image,0,1)
    while np.sum(image[-1]) == 0:
        image = image[:-1]
    while np.sum(image[:,-1]) == 0:
        image = np.delete(image,-1,1)

    rows, cols = image.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols * factor))
        image = cv2.resize(image, (cols,rows), interpolation=cv2.INTER_AREA)
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows * factor))
        image = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _ret, image = cv2.threshold(image, 128, 255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Add padding to make 28x28
    colsPadding = int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0))
    rowsPadding = int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0))
    image = np.pad(image, (rowsPadding, colsPadding), 'constant')

    return image

if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        print("Usage: python preprocess.py <path>")
        exit(0)

    path = sys.argv[1]
    image = preprocess_image(path)

    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()
