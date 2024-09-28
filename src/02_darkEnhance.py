from __future__ import annotations
from typing import Callable

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
    ResultImage(output_name="02_global", method="global_histogram", kernel_size=0),
    # ResultImage(output_name="03_local_3x3", method="test", kernel_size=3),
    # ResultImage(output_name="04_local_7x7", method="test", kernel_size=7),
    # ResultImage(output_name="05_local_11x11", method="test", kernel_size=11),
    # ResultImage(output_name="06_gamma_5x5", method="test", kernel_size=5),
    # ResultImage(output_name="07_gamma_9x9", method="test", kernel_size=9),
    # ResultImage(output_name="08_gamma_15x15", method="test", kernel_size=15),
]
# endregion


def save_histogram(image: MatLike, request: ResultImage) -> None:
    histogram: MatLike = cv.calcHist(
        [image],
        channels=[0],
        mask=None,
        histSize=[max_intensity + 1],
        ranges=[0, max_intensity + 1],
    )
    intensity_range: NDArray = 0.5 + np.arange(max_intensity + 1)

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.bar(x=intensity_range, height=histogram.flatten(), width=1)

    plt.savefig(f"{result_folder}/{request.output_name}_histogram.jpg")


def global_histogram(image: MatLike) -> NDArray[np.uint8]:
    histogram: MatLike = (
        cv.calcHist(
            [image],
            channels=[0],
            mask=None,
            histSize=[max_intensity + 1],
            ranges=[0, max_intensity + 1],
        )
        / image.size
    )
    cdf: NDArray = histogram.cumsum()

    new_image: NDArray = np.zeros_like(image, np.float16)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            intensity: np.uint8 = image[i, j]
            new_image[i, j] = cdf[intensity] * max_intensity

    return new_image.astype(np.uint8)


if __name__ == "__main__":
    for request in result_set:
        print(f"Operating on request: {request.output_name} with {request.method}...")
        image: MatLike = cv.imread(filename=fillament, flags=cv.IMREAD_GRAYSCALE)
        if request.method == "global_histogram":
            image = global_histogram(image)
        print(f"Saving result...")
        cv.imwrite(filename=f"{result_folder}/{request.output_name}.jpg", img=image)
        save_histogram(image, request)
        print(f"Request is done.\n")
