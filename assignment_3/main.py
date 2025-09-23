import cv2
import cv2 as cv
import numpy as np
from typing import Literal


def sobel_edge_detection(image) -> None:
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(image_grey, (3, 3), 0)
    sobel = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=1)

    sobel = (255 * sobel).clip(0, 255).astype(np.uint8)
    cv.imwrite('solutions/Sobel_edge_detection.png', sobel)


def canny_edge_detection(image, threshold1=50, threshold2=50) -> None:
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(image_grey, (3, 3), 0)

    canny = cv.Canny(img_blur, threshold1, threshold2)
    cv.imwrite('solutions/Canny_edge_detection.png', canny)

def template_match(image, template) -> None:
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template_grey = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    w, h, _ = template.shape

    res = cv.matchTemplate(image_grey, template_grey, cv.TM_CCOEFF_NORMED)

    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv.imwrite('solutions/template_match.png', image)

def resize(image, scale_factor: int, up_or_down: Literal["up" , "down"]) -> None:
    rows, cols, _channels = map(int, image.shape)

    if up_or_down == "up":
        image = cv.pyrUp(image, dstsize=(scale_factor * cols, scale_factor * rows))

    elif up_or_down == 'down':
        image = cv.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))

    else:
        assert False, "Invalid up_or_down parameter"

    cv.imwrite('solutions/resize.png', image)


if __name__ == '__main__':
    img = cv.imread('lambo.png')

    sobel_edge_detection(img)
    canny_edge_detection(img)

    shapes = cv.imread('shapes-1.png')
    shapes_template =cv.imread('shapes_template.jpg')
    template_match(shapes, shapes_template)

    resize(img, 2, 'down')

