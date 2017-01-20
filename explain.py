
import argparse
import cv2
import scipy.misc

from process_image import read_image, get_slice_masks,get_best_sliced_image
from predict import predict, load_model


n_rows = 33

if __name__=='__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-m", "--model", required=True, help="Path to the vgg16 weights .h5-file ")
    ap.add_argument('-o', '--output', default='out.jpg', help='Path for image output')
    args = vars(ap.parse_args())

    print 'Loading image, producing slices'
    image = read_image(args['image'])
    masks = get_slice_masks(image)

    print 'Loading model'
    model = load_model(args['model'])

    image_classification =  predict(model, image, order=True)[0]
    print 'Classification with highest probability: %s (p=%.2f)' % (image_classification[0], image_classification[1])
    print 'Identifying best image'
    best_image = get_best_sliced_image(image, model, masks, n_rows=n_rows)
    scipy.misc.imsave(args['output'], best_image)
