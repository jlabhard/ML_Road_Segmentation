''' General imports'''
import os
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
import re
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from PIL import Image

''' Global variables'''

'''Directories'''
OVERALL_DIRECTORY = ''
MODEL_PATH = OVERALL_DIRECTORY + 'unet_model.h5'
DATA_DIR = OVERALL_DIRECTORY + 'Datasets/test_set_images/'
SUBMISSION_FILENAME = OVERALL_DIRECTORY + "submission.csv"

'''Parameters'''
FOREGROUND_THRESHOLD = 0.25
PIXEL_DEPTH = 255
PATCH_SIZE = 16
IMAGE_SIZE = 608
MODEL_INPUT_SIZE = 400


'''
Classify the images by patch.
    - patch: subset of a image of size [PATCH_SIZE x PATCH_SIZE]
return:
    The classification label of the patch, 0 or 1.
'''


def patch_to_label(patch):
    df = np.mean(patch)
    if df > FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0


'''
Compute the prediction of an image. Since the model takes images of size 400x400
as input and our test images are of size 608x608, we spit the image in 4 images
of size 400x400 and then combine them into one prediction.
PARAMETERS:
    - img: the image to predict of size (608x608)
    - model: model saved at MODEL_PATH
RETURN:
    The prediction of the image
'''


def large_predict(img, model):
    # Split image in four 400x400 images to adapt to input size of model
    imgs = np.array([img[:MODEL_INPUT_SIZE, :MODEL_INPUT_SIZE],
                     img[:MODEL_INPUT_SIZE, -MODEL_INPUT_SIZE:],
                     img[-MODEL_INPUT_SIZE:, :MODEL_INPUT_SIZE],
                     img[-MODEL_INPUT_SIZE:, -MODEL_INPUT_SIZE:]])

    predictions = model.predict(imgs)

    # Combine the four predictions to generate prediction of the image
    prediction = np.empty((IMAGE_SIZE, IMAGE_SIZE, 1))
    prediction[:MODEL_INPUT_SIZE, :MODEL_INPUT_SIZE] = predictions[0]
    prediction[:MODEL_INPUT_SIZE, -MODEL_INPUT_SIZE:] = predictions[1]
    prediction[-MODEL_INPUT_SIZE:, :MODEL_INPUT_SIZE] = predictions[2]
    prediction[-MODEL_INPUT_SIZE:, -MODEL_INPUT_SIZE:] = predictions[3]

    return prediction


'''
Converts images into a submission file in csv format.
PARAMETERS:
    - directory: directory containing the predictions.
    - image_filenames: contains the prediction of all the test images.
'''


def masks_to_submission(directory, *image_filenames):
    with open(SUBMISSION_FILENAME, "w") as f:
        f.write("id,prediction\n")
        for fn in image_filenames[0:]:
            f.writelines("{}\n".format(s)
                         for s in mask_to_submission_strings(fn, directory))


'''
Reads a single image and outputs the strings that should go into the submission file
PARAMETERS:
    - image_filename: the prediction of a single image.
    - directory: directory containing the predictions.
YIELD:
    output a string format corresponding to the classification label for a given patch and for a given image.
'''


def mask_to_submission_strings(image_filename, directory):
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(directory + image_filename)
    for j in range(0, im.shape[1], PATCH_SIZE):
        for i in range(0, im.shape[0], PATCH_SIZE):
            patch = im[i: i + PATCH_SIZE, j: j + PATCH_SIZE]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


'''
Min max scaling of a image
PARAMETER:
    - img: image where to do the scaling
RETURN:
    The normalized image
'''


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


'''
Load every images containes in images_path and convert it into numpy arrays.
PARAMETERS:
    - images_path: path of the directory containing images you want to load.
    - groundtruth: boolean patameter to know if the image is a groundtruth or not.
RETURN:
    Array containing every array representation of the images contained in images_path
'''


def extract_images(images_path, groundtruth=False):
    imgs = []
    for root, _, files in os.walk(images_path, topdown=True):
        for fn in sorted(files):
            image_filename = os.path.join(root, fn)
            if os.path.isfile(image_filename):
                print('Loading ' + image_filename)
                if (groundtruth == True):
                    img = load_img(
                        image_filename, color_mode="grayscale")
                else:
                    img = load_img(image_filename)
                imgs.append(img_to_array(img) / 255)
            else:
                print('File ' + image_filename + ' does not exist')
    return np.array(imgs)


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


if __name__ == '__main__':

    model = tf.keras.models.load_model(MODEL_PATH)

    data = extract_images(images_path=DATA_DIR)

    image_filenames = []
    prediction_testing_dir = OVERALL_DIRECTORY + 'predictions_testing/'
    if not os.path.isdir(prediction_testing_dir):
        os.mkdir(prediction_testing_dir)

    for i in range(data.shape[0]):
        prediction = large_predict(data[i], model)
        pimg = np.squeeze(prediction).round()
        oimg = make_img_overlay(data[i], pimg)
        oimg.save(prediction_testing_dir + "overlay_" + str(i + 1) + ".png")
        pimg = img_float_to_uint8(pimg)
        Image.fromarray(pimg).save(
            prediction_testing_dir + "prediction_" + str(i + 1) + ".png")

        image_filename = "prediction_" + str(i + 1) + ".png"
        image_filenames.append(image_filename)

    masks_to_submission(prediction_testing_dir, *image_filenames)
