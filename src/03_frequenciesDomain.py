import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from cv2.typing import MatLike
from numpy.typing import NDArray


class ResultImage:
    def __init__(self, filename: str, filter_name: str, is_low_pass: bool):
        self.filename = filename
        self.filter_name = filter_name
        self.is_low_pass = is_low_pass


# region config
NOISY_TOM_JERRY = "Noisy_Tom_Jerry"
NOISY_WHIRLPOOL = "Noisy_whirlpool"
NOISY_GALAXY_3 = "Noisy_galaxy3"

IDEAL_FILTER = "IDEAL_FILTER"
GAUSSIAN_FILTER = "GAUSSIAN_FILTER"

LOW_PASS: bool = True
HIGH_PASS: bool = False

result_set: list[ResultImage] = [
    # 1.a
    ResultImage(NOISY_TOM_JERRY, IDEAL_FILTER, LOW_PASS),
    ResultImage(NOISY_WHIRLPOOL, IDEAL_FILTER, LOW_PASS),
    ResultImage(NOISY_GALAXY_3, IDEAL_FILTER, LOW_PASS),
    # 1.b
    ResultImage(NOISY_TOM_JERRY, IDEAL_FILTER, HIGH_PASS),
    ResultImage(NOISY_WHIRLPOOL, IDEAL_FILTER, HIGH_PASS),
    ResultImage(NOISY_GALAXY_3, IDEAL_FILTER, HIGH_PASS),
    # 2.a
    ResultImage(NOISY_TOM_JERRY, GAUSSIAN_FILTER, LOW_PASS),
    ResultImage(NOISY_WHIRLPOOL, GAUSSIAN_FILTER, LOW_PASS),
    ResultImage(NOISY_GALAXY_3, GAUSSIAN_FILTER, LOW_PASS),
    # 2.b
    ResultImage(NOISY_TOM_JERRY, GAUSSIAN_FILTER, HIGH_PASS),
    ResultImage(NOISY_WHIRLPOOL, GAUSSIAN_FILTER, HIGH_PASS),
    ResultImage(NOISY_GALAXY_3, GAUSSIAN_FILTER, HIGH_PASS),
    # 3
    ResultImage(NOISY_GALAXY_3, IDEAL_FILTER, LOW_PASS),
]
# endregion


def scale_intensity(
    image: NDArray[np.float32],
    min: np.float32 = np.float32(0.0),
    max: np.float32 = np.float32(255.0),
) -> NDArray[np.float32]:
    image = (image - np.min(image) + min).astype(np.float32)
    image = image * max / np.max(image).astype(np.float32)

    return image


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
    low_pass_filter_mask: NDArray[np.float32] = np.exp(
        -distance_square / (2 * cutoff**2)
    )

    return low_pass_filter_mask if is_low_pass else 1 - low_pass_filter_mask


def dft_and_spectrum(
    image: NDArray[np.uint8],
) -> tuple[NDArray[np.complex128], NDArray[np.uint8]]:
    dft: NDArray[np.complex128] = np.fft.fft2(image)
    dft = np.fft.fftshift(dft)
    spectrum: NDArray[np.float32] = np.log(np.abs(dft)).astype(np.float32)
    spectrum = scale_intensity(spectrum)
    return dft, spectrum.astype(np.uint8)


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
    main()
