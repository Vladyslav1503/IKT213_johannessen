import cv2 as cv


def print_image_information(image):
    img = cv.imread(image)
    width, height, channels = img.shape
    print(f"width: {width}")
    print(f"height: {height}")
    print(f"Channels: {channels}")
    print(f"size {img.size}")
    print(f"data type: {img.dtype}")


if __name__ == '__main__':
    print_image_information("lena.png")

