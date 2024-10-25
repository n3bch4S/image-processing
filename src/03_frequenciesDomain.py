from turtle import distance
import cv2 as cv
from matplotlib.figure import Figure
from networkx import center
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
k1: float = 0.04
k2: float = 0.3
e: float = 5.0

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


def show(image: NDArray[np.uint8]) -> None:
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10, 10), layout="constrained")
    axs[0, 0].imshow(image, cmap="gray")
    plt.show()


def center_distance_map(shape: tuple[int, int]) -> NDArray[np.float32]:
    rows, cols = shape
    x: NDArray[np.float32] = np.linspace(
        start=-cols // 2 + 1, stop=cols // 2, num=cols, dtype=np.float32
    )
    y: NDArray[np.float32] = np.linspace(
        start=-rows // 2 + 1, stop=rows // 2, num=rows, dtype=np.float32
    )

    X, Y = np.meshgrid(x, y)
    return np.sqrt(X**2 + Y**2, dtype=np.float32)


def ideal_filter(
    shape: tuple[int, int], radius: int, is_low_pass: bool
) -> NDArray[np.float32]:
    distance: NDArray[np.float32] = center_distance_map(shape)

    filter_mask: NDArray[np.float32] = np.zeros(shape, dtype=np.float32)
    filter_mask[distance <= radius if is_low_pass else distance > radius] = 1

    return filter_mask


def gaussian_filter(
    shape: tuple[int, int], cutoff: int, is_low_pass: bool
) -> NDArray[np.float32]:
    distance_square: NDArray[np.float32] = center_distance_map(shape) ** 2

    return np.exp(-distance_square / (2 * cutoff**2))


def main() -> None:
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


if __name__ == "__main__":
    ideal_mask: NDArray[np.float32] = gaussian_filter(
        shape=(100, 100), cutoff=10, is_low_pass=True
    )
    filter: NDArray[np.uint8] = (ideal_mask * 255).astype(np.uint8)
    print(filter)
    show(filter)
