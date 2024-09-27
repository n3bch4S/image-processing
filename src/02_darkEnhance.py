from __future__ import annotations

import cv2 as cv
import numpy as np

from cv2.typing import MatLike
from numpy.typing import NDArray


class ResultImage:
    def __init__(self, output_name: str, method: str, kernel_size: int):
        self.output_name = output_name
        self.method = method
        self.kernel_size = kernel_size


# region config
k0: float = 1
k1: float = 1
k2: float = 1
e: float = 1
gamma: float = 1
max_intensity: int = 255
fillament: str = "../assets/assgn_02/Filament.jpg"
result_folder: str = "../result/assgn_02"
result_set: list[ResultImage] = [
    ResultImage(output_name="01_original", method="", kernel_size=0),
    ResultImage(output_name="03_global", method="test", kernel_size=0),
    ResultImage(output_name="05_local_3x3", method="test", kernel_size=3),
    ResultImage(output_name="07_local_7x7", method="test", kernel_size=7),
    ResultImage(output_name="09_local_11x11", method="test", kernel_size=11),
    ResultImage(output_name="11_gamma_5x5", method="test", kernel_size=5),
    ResultImage(output_name="13_gamma_9x9", method="test", kernel_size=9),
    ResultImage(output_name="15_gamma_15x15", method="test", kernel_size=15),
]
# endregion


def global_histogram(image: MatLike):
    histogram: MatLike = cv.calcHist(
        [image], channels=[0], mask=None, histSize=[256], ranges=[0, 256]
    )
    histogram = histogram / image.size

    cdf: NDArray = histogram.cumsum()
    print(cdf.shape)
    # # Step 5: Map the intensity levels
    # equalized_image = (
    # np.interp(image.flatten(), np.arange(max_intensity + 1), cdf)
    #     .reshape(image.shape)
    #     .astype(np.uint8)
    # )

    # return equalized_image
    return image


if __name__ == "__main__":
    image: MatLike = cv.imread(filename=fillament, flags=cv.IMREAD_GRAYSCALE)
    global_histogram(image)
    cv.imwrite(filename=f"{result_folder}/gray.jpg", img=image)
    # print(type((image[2, 1] * 7.9).astype(np.uint8)))
