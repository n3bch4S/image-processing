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
k0: float = 0.4
k1: float = 0.01
k2: float = 0.3
e: float = 6.0

local_gamma_bias: float = 0.4

max_intensity: int = 255
fillament: str = "../assets/assgn_02/Filament.jpg"
result_folder: str = "../result/assgn_02"
result_set: list[ResultImage] = [
    ResultImage(output_name="01_original", method="", kernel_size=0),
    ResultImage(output_name="02_global", method="global_histogram", kernel_size=0),
    ResultImage(output_name="03_local_3x3", method="local_histogram", kernel_size=3),
    ResultImage(output_name="04_local_7x7", method="local_histogram", kernel_size=7),
    ResultImage(output_name="05_local_11x11", method="local_histogram", kernel_size=11),
    ResultImage(
        output_name="06_gamma_5x5", method="local_gamma_correction", kernel_size=5
    ),
    ResultImage(
        output_name="07_gamma_9x9", method="local_gamma_correction", kernel_size=9
    ),
    ResultImage(
        output_name="08_gamma_15x15", method="local_gamma_correction", kernel_size=15
    ),
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


def local_histogram(image: MatLike, kernel_size: int) -> NDArray[np.uint8]:
    global_mean: np.float32 = np.mean(image.astype(np.float32))
    global_std: np.float32 = np.std(image.astype(np.float32))

    pad_width: int = kernel_size // 2
    padded_image: MatLike = cv.copyMakeBorder(
        image,
        top=pad_width,
        bottom=pad_width,
        left=pad_width,
        right=pad_width,
        borderType=cv.BORDER_REFLECT,
    )

    new_image: NDArray = np.zeros_like(image, np.float16)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_region: NDArray = padded_image[
                i : i + kernel_size, j : j + kernel_size
            ]
            local_mean: np.float32 = np.mean(local_region.astype(np.float32))
            local_std: np.float32 = np.std(local_region.astype(np.float32))

            is_below_global_mean: bool = bool(local_mean <= k0 * global_mean)
            is_between_global_std: bool = bool(
                k1 * global_std <= local_std <= k2 * global_std
            )
            new_image[i, j] = (
                np.clip(e * image[i, j], a_min=0, a_max=255)
                if is_below_global_mean and is_between_global_std
                else image[i, j]
            )

    return new_image.astype(np.uint8)


def local_gamma_correction(image: MatLike, kernel_size: int) -> NDArray[np.uint8]:
    global_mean: np.float16 = np.mean(image.astype(np.float16))

    pad_width: int = kernel_size // 2
    padded_image: MatLike = cv.copyMakeBorder(
        image,
        top=pad_width,
        bottom=pad_width,
        left=pad_width,
        right=pad_width,
        borderType=cv.BORDER_REFLECT,
    )

    new_image: NDArray = np.zeros_like(image, np.float16)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_region: NDArray = padded_image[
                i : i + kernel_size, j : j + kernel_size
            ]
            local_mean: np.float16 = np.mean(local_region.astype(np.float16))

            local_gamma: np.float16 = (
                1 - (global_mean - local_mean) / (global_mean) + local_gamma_bias
            )
            new_image[i, j] = (
                np.power(image[i, j] / max_intensity, local_gamma) * max_intensity
            )

    return new_image.astype(np.uint8)


if __name__ == "__main__":
    for request in result_set:
        print(f"Operating on request: {request.output_name} with {request.method}...")

        image: MatLike = cv.imread(filename=fillament, flags=cv.IMREAD_GRAYSCALE)

        if request.method == "global_histogram":
            image = global_histogram(image)
        elif request.method == "local_histogram":
            image = local_histogram(image, request.kernel_size)
        elif request.method == "local_gamma_correction":
            image = local_gamma_correction(image, request.kernel_size)

        print(f"Saving result...")

        cv.imwrite(filename=f"{result_folder}/{request.output_name}.jpg", img=image)
        save_histogram(image, request)

        print(f"Request is done.\n")
