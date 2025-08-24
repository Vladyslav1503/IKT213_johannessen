import cv2 as cv
from pathlib import Path

class WebCamInfo:
    width : int
    height : int
    fps : int


def get_webcam_info() -> WebCamInfo:
    camera = cv.VideoCapture(0)
    cam_info = WebCamInfo()

    cam_info.width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
    cam_info.height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))

    cam_info.fps = camera.get(cv.CAP_PROP_FPS)

    return cam_info


if __name__ == '__main__':
    web_cam_info : WebCamInfo = get_webcam_info()

    Path("solutions").mkdir(parents=True, exist_ok=True)

    with open("solutions/camera_outputs.txt", "w") as outfile:
        outfile.write(f"Width: {web_cam_info.width}\n")
        outfile.write(f"height: {web_cam_info.height}\n")
        outfile.write(f"fps: {web_cam_info.fps}\n")

