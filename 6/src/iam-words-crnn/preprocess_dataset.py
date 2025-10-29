import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

target_height = 32
target_chunk_width = 8
target_chunks = 45
target_width = target_chunks * target_chunk_width

num_elems_guaranteed = 33000

train_portion = 0.8

seed = 13548613

def preprocess_dataset():
    x_images = np.empty((num_elems_guaranteed, target_height, target_width), dtype=np.uint8)
    y_labels = np.empty((num_elems_guaranteed,), dtype="<U32")
    alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    lines_read = 0
    images_preprocessed = 0
    with open("dataset-raw/words.txt", 'r') as txtfile:
        for (line_idx, line) in enumerate(txtfile):
            if line.startswith("#"):
                continue
            
            lines_read += 1

            line_split = line.strip().split(' ')
            path, status, graylevel, word = line_split[0], line_split[1], int(line_split[2]), line_split[8]

            if status != "ok":
                continue
            if len(word) < 2:
                continue

            path_split = path.split('-')
            dir, subdir, filename = path_split[0], '-'.join([path_split[0], path_split[1]]), f"{path}.png"
            path = f"dataset-raw/words/{dir}/{subdir}/{filename}"

            try:
                image = preprocess_word_png(path, graylevel)

                x_images[images_preprocessed] = image
                y_labels[images_preprocessed] = word

                images_preprocessed += 1

                for ch in word:
                    alphabet.add(ch)

                if line_idx % 100 == 0:
                    print(f"Line {line_idx}: {images_preprocessed} images processed")

                if images_preprocessed >= num_elems_guaranteed:
                    break

            except AssertionError:
                print(f"Scipped too wide picture {filename}")
                continue
            except cv2.error:
                print(f"OpenCV error while processing {filename}")
                continue

    num_elems = y_labels.size

    print(f"Lines read: {lines_read}; Images preprocessed: {num_elems}")
    print("Shuffling...")

    np.random.seed(seed)
    indices = np.arange(num_elems)
    # np.random.shuffle(indices)

    x_images = x_images[indices]
    y_labels = y_labels[indices]

    train_count = round(train_portion * num_elems)

    x_train = x_images[:train_count]
    y_train = y_labels[:train_count]
    x_test = x_images[train_count:]
    y_test = y_labels[train_count:]

    x_train.tofile("dataset/train-images.idx3-ubyte")
    y_train.tofile("dataset/train-labels.idx1-U32")
    x_test.tofile("dataset/t10k-images.idx3-ubyte")
    y_test.tofile("dataset/t10k-labels.idx1-U32")

    alphabet = "".join(sorted(list(alphabet)))
    with open("dataset/alphabet.txt", "w") as alphabet_file:
        alphabet_file.write(alphabet)

    print("Done!")


def preprocess_word_png(path, graylevel=None):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    rows, cols = image.shape
    factor = target_height / rows
    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    if graylevel is None:
        _ret, image = cv2.threshold(image, 0, 255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _ret, image = cv2.threshold(image, graylevel, 255, type=cv2.THRESH_BINARY)
    image = cv2.bitwise_not(image)

    rows, cols = image.shape
    assert cols < target_width, f"See {path}"
    colsPadding = int(math.ceil((target_width - cols) / 2.0)), int(math.floor((target_width - cols) / 2.0))
    image = np.pad(image, ((0, 0), colsPadding), 'constant')

    return image


if __name__ == "__main__":
    preprocess_dataset()
