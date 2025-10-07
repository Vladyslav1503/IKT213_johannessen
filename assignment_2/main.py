import cv2
import cv2 as cv
import numpy as np


def padding(image, border_width=100):
    reflect = cv.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv.BORDER_REFLECT)

    return reflect


def crop(image, x_0, x_1, y_0, y_1):
    return image[y_0:y_1, x_0:x_1]


def resize(image, width, height):
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)


def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    emptyPictureArray = np.zeros((width, height, channels), dtype=np.uint8)

    i = 0
    for pixel in image:
        emptyPictureArray[i] = pixel
        i += 1

    return emptyPictureArray


def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def hsv(image):
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def hue_shifted(image, emptyPictureArray, hue):
    MAX_VALUE = 255

    height, width, channels = image.shape
    emptyPictureArray = np.zeros((width, height, channels), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            for k in range(channels):
                value = int(image[i, j, k]) + hue
                value = max(0, min(MAX_VALUE, value))
                emptyPictureArray[i, j, k] = value

    return emptyPictureArray


def smoothing(image):
    return cv.GaussianBlur(image, (15, 15), cv.BORDER_DEFAULT)


def rotation(image, rotation_angle):
    match rotation_angle:
        case 90:
            return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        case 180:
            return cv.rotate(image, cv.ROTATE_180)
        case 270:
            return cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
        case _:
            raise ValueError("Only 90, 180, or 270 degrees are supported.")


if __name__ == "__main__":
    img = cv.imread('lena-2.png')

    width = img.shape[0]
    height = img.shape[1]

    padding_image = padding(img, border_width=100)
    cv.imwrite('solutions/padding.png', padding_image)

    crop_image = crop(img, 80, width, 80, height)
    crop_image_width, crop_image_height, _ = crop_image.shape
    crop_image = crop(crop_image, 0, crop_image_width - 130, 0, crop_image_height - 130)
    cv.imwrite('solutions/cropped.png', crop_image)

    resized_image = resize(img, 200, 200)
    cv.imwrite('solutions/resized.png', resized_image)

    copy_image = copy(img, [])
    cv.imwrite('solutions/copy.png', copy_image)

    grey_image = grayscale(img)
    cv.imwrite('solutions/grayscale.png', grey_image)

    hsv_image = hsv(img)
    cv.imwrite('solutions/hsv.png', hsv_image)

    hoe_shifted_image = hue_shifted(img, [], 50)
    cv.imwrite('solutions/hoe_shifted.png', hoe_shifted_image)

    smoothing_image = smoothing(img)
    cv.imwrite('solutions/smoothing.png', smoothing_image)

    rotated_image_90 = rotation(img, 90)
    rotated_image_180 = rotation(img, 180)
    cv.imwrite('solutions/rotated_image_90.png', rotated_image_90)
    cv.imwrite('solutions/rotated_image_180.png', rotated_image_180)
