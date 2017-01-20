from skimage.segmentation import slic
from skimage.util import img_as_float
import cv2
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV

from predict import predict, load_model



def read_image(filename):
    return cv2.resize(cv2.imread(filename), (224, 224))#.astype(np.float32)


def get_slice_masks(image, n_segments=10):
    segments = slic(img_as_float(image), n_segments=n_segments, sigma=5)
    masks = []
    # loop over the unique segment values
    for (i, segVal) in enumerate(np.unique(segments)):
        # construct a mask for the segment
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255
        masks.append(mask)
    return masks
def combine_masks(masks):
    if len(masks) == 1:
        return masks[0]
    else:
        return reduce(lambda x, y: x + y, masks)


def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def get_x_y_pair(masks, img, model, class_index=0):
    ## get_random_mask
    rmask, features = get_random_mask(masks)
    ## apply_mask
    new_img = apply_mask(img, rmask)
    ## predict
    prediction = predict(model, new_img, order=False)
    ## get prediction of class_index
    prediction = prediction[class_index][1]
    ## return feature and class prob
    return(features, prediction)

def get_random_mask(masks):
    n_pieces = max(1 + np.random.poisson(2, 3)[0], len(masks))
    random_index = np.random.choice(range(0, len(masks)), n_pieces)
    random_masks = [masks[i] for i in random_index]
    feature_vector = np.zeros(len(masks))
    feature_vector[random_index] = 1
    combined_mask = combine_masks([masks[i] for i in random_index])
    return (combined_mask, feature_vector)

def get_best_sliced_image(image, model, masks, n_rows=10):
    max_index = np.argmax([x[1] for x in predict(model, image, order=False)])
    xy_pairs = [get_x_y_pair(masks, image, model, class_index=max_index) for i in range(0, n_rows)]
    X = pd.DataFrame([x[0] for x in xy_pairs])
    y = pd.Series([x[1] for x in xy_pairs])
    mod = LassoCV()
    mod.fit(X, y)
    features_selection = np.where(mod.coef_ > 0)[0]
    features_selection
    best_mask = combine_masks([masks[i] for i in features_selection])
    best_image = apply_mask(image, best_mask)
    return best_image
