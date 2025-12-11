import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from iam_sentences_crnn.preprocess_dataset import target_height, target_width
import math
import torch
import matplotlib.pyplot as plt

def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _ret, image = cv2.threshold(image, 0, 255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    image = cv2.bitwise_not(image)

    return image

def rows_avg(image, sigma=10.0):
    avg = np.mean(image, axis=1, dtype=np.float32)
    avg_smoothed = gaussian_filter1d(avg, sigma=sigma)
    
    return avg_smoothed

def determine_avg_threshold(avg):
    avg_normalized = ((avg - avg.min()) / (avg.max() - avg.min()) * 255).astype(np.uint8)
    threshold, _ = cv2.threshold(avg_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_original = threshold / 255 * (avg.max() - avg.min()) + avg.min()
    
    adjusted_threshold = threshold_original * 0.15

    return adjusted_threshold

def group_continuous_regions(avg):
    threshold = determine_avg_threshold(avg)

    bool_mask = avg > threshold
    
    # Находим изменения состояний
    changes = np.diff(bool_mask.astype(int))
    change_indices = np.where(changes != 0)[0] + 1
    
    # Создаем сегменты
    segments = []
    start = 0
    
    for change_point in change_indices:
        segments.append((start, change_point, bool_mask[start]))
        start = change_point
    segments.append((start, len(avg), bool_mask[start]))
    
    # Группируем по типу
    bright_regions = [(s, e) for s, e, is_bright in segments if is_bright]
    dark_regions = [(s, e) for s, e, is_bright in segments if not is_bright]
    
    return bright_regions, dark_regions

def assemble_lines(dark_regions):
    num_darks = np.size(dark_regions, axis=0)

    lines = []
    for i in range(num_darks - 1):
        lines.append((dark_regions[i][0], dark_regions[i+1][1]))

    return lines

def line_to_image(image, line):
    return image[line[0]:line[1], :]

def trim_line_image(line_image):
    avg = np.mean(line_image, axis=1)
    mask = avg > 0.0

    res = line_image[mask, :]

    while np.sum(res[:,0]) == 0:
        res = np.delete(res,0,1)
    while np.sum(res[:,-1]) == 0:
        res = np.delete(res,-1,1)
    return res

def resize_line_image_for_model_input(image):
    rows, cols = image.shape
    factor = target_height / rows
    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    rows, cols = image.shape
    if cols > target_width:
        image = cv2.resize(image, (target_width, rows), interpolation=cv2.INTER_AREA)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    elif cols < target_width:
        colsPadding = int(math.ceil((target_width - cols) / 2.0)), int(math.floor((target_width - cols) / 2.0))
        image = np.pad(image, ((0, 0), colsPadding), 'constant')

    return image

def generate_model_input(image_path):
    image = preprocess_image(image_path)
    avg = rows_avg(image)

    bright_regions, dark_regions = group_continuous_regions(avg)
    lines = assemble_lines(dark_regions)
    
    def complete_line_transform(line):
        img = line_to_image(image, line)
        img = trim_line_image(img)
        img = resize_line_image_for_model_input(img)

        return img
    
    lines = np.array([complete_line_transform(line) for line in lines])
    lines = np.array([np.array([line]) for line in lines])
    lines = torch.Tensor(lines)

    return lines
