import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from cv2.typing import MatLike
from numpy.typing import NDArray


class ResultImage:
    def __init__(self, filename: str, filter_name: str, pass_type: str):
        self.filename = filename
        self.filter_name = filter_name
        self.pass_type = pass_type


# region config
NOISY_TOM_JERRY = "Noisy_Tom_Jerry"
NOISY_WHIRLPOOL = "Noisy_whirlpool"
NOISY_GALAXY_3 = "Noisy_galaxy3"

IDEAL_FILTER = "IDEAL_FILTER"
GAUSSIAN_FILTER = "GAUSSIAN_FILTER"

LOW_PASS: str = "LOW_PASS"
HIGH_PASS: str = "HIGH_PASS"

CUTOFFS: list[int] = [10, 50, 100]

RESULT_SET: list[ResultImage] = [
    # 1.a
    ResultImage(NOISY_TOM_JERRY, IDEAL_FILTER, LOW_PASS),
    # ResultImage(NOISY_WHIRLPOOL, IDEAL_FILTER, LOW_PASS),
    # ResultImage(NOISY_GALAXY_3, IDEAL_FILTER, LOW_PASS),
    # 1.b
    ResultImage(NOISY_TOM_JERRY, IDEAL_FILTER, HIGH_PASS),
    # ResultImage(NOISY_WHIRLPOOL, IDEAL_FILTER, HIGH_PASS),
    # ResultImage(NOISY_GALAXY_3, IDEAL_FILTER, HIGH_PASS),
    # 2.a
    ResultImage(NOISY_TOM_JERRY, GAUSSIAN_FILTER, LOW_PASS),
    # ResultImage(NOISY_WHIRLPOOL, GAUSSIAN_FILTER, LOW_PASS),
    # ResultImage(NOISY_GALAXY_3, GAUSSIAN_FILTER, LOW_PASS),
    # 2.b
    ResultImage(NOISY_TOM_JERRY, GAUSSIAN_FILTER, HIGH_PASS),
    # ResultImage(NOISY_WHIRLPOOL, GAUSSIAN_FILTER, HIGH_PASS),
    # ResultImage(NOISY_GALAXY_3, GAUSSIAN_FILTER, HIGH_PASS),
    # 3
    # ResultImage(NOISY_GALAXY_3, IDEAL_FILTER, LOW_PASS),
]
# endregion


def scale_intensity(
    image: NDArray[np.float32],
    min: int = 0,
    max: int = 255,
) -> NDArray[np.float32]:
    image = image - np.min(image) + min
    image = image * max / np.max(image)

    return image.astype(np.float32)


def plot_image(image: NDArray[np.uint8], name: str, *index: int) -> None:
    plt.subplot(*index)
    plt.imshow(image, "gray")
    plt.title(name)
    plt.xticks([])
    plt.yticks([])


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
    shape: tuple[int, int], radius: int, pass_type: str
) -> NDArray[np.float32]:
    distance: NDArray[np.float32] = center_distance_map(shape)

    filter_mask: NDArray[np.float32] = np.zeros(shape, dtype=np.float32)
    filter_mask[distance <= radius if pass_type == LOW_PASS else distance > radius] = 1

    return filter_mask


def gaussian_filter(
    shape: tuple[int, int], cutoff: int, pass_type: str
) -> NDArray[np.float32]:
    distance_square: NDArray[np.float32] = center_distance_map(shape) ** 2
    low_pass_filter_mask: NDArray[np.float32] = np.exp(
        -distance_square / (2 * cutoff**2), dtype=np.float32
    )

    return low_pass_filter_mask if pass_type == LOW_PASS else (1 - low_pass_filter_mask)


def spectrum_of(dft: NDArray[np.complex128]) -> NDArray[np.uint8]:
    spectrum: NDArray[np.float32] = np.log(np.abs(dft) + 1, dtype=np.float32)
    return scale_intensity(spectrum).astype(np.uint8)


def dft_and_spectrum(
    image: NDArray[np.uint8],
) -> tuple[NDArray[np.complex128], NDArray[np.uint8]]:
    dft: NDArray[np.complex128] = np.fft.fft2(image)
    dft = np.fft.fftshift(dft)
    return dft, spectrum_of(dft)


def spatial_from(dft: NDArray[np.complex128]) -> NDArray[np.uint8]:
    ifft: NDArray[np.complex128] = np.fft.ifft2(np.fft.ifftshift(dft))
    real_part: NDArray[np.float32] = np.real(ifft).astype(np.float32)

    return scale_intensity(real_part).astype(np.uint8)


def main() -> None:
    for request in RESULT_SET:
        print(f"Operating on request: {request.filename} with {request.filter_name}...")

        path: str = f"../assets/assignment_03/{request.filename}.jpg"
        image: NDArray[np.uint8] = cv.imread(path, cv.IMREAD_GRAYSCALE).astype(np.uint8)
        dft, spectrum = dft_and_spectrum(image)

        plt.figure(figsize=(10, 9))
        plot_image(image, f"Original {request.filename}", 4, 3, 1)
        plot_image(spectrum, f"Spectrum {request.filename}", 4, 3, 2)

        for i in range(len(CUTOFFS)):
            cutoff: int = CUTOFFS[i]
            if request.filter_name == IDEAL_FILTER:
                filter: NDArray[np.float32] = ideal_filter(
                    spectrum.shape, cutoff, request.pass_type
                )
            else:
                filter: NDArray[np.float32] = gaussian_filter(
                    spectrum.shape, cutoff, request.pass_type
                )

            filtered_dft: NDArray[np.complex128] = (dft * filter).astype(np.complex128)
            spectrum = spectrum_of(filtered_dft)
            image = spatial_from(filtered_dft)

            cell: int = 3 * (i + 1)
            plot_image(
                scale_intensity(filter).astype(np.uint8),
                f"{request.pass_type} {request.filter_name} cutoff={cutoff}",
                4,
                3,
                cell + 1,
            )
            plot_image(spectrum, f"Filtered Spectrum", 4, 3, cell + 2)
            plot_image(image, f"After Filtered", 4, 3, cell + 3)

        print(f"Saving result...")
        plt.savefig(
            f"../result/assignment_03/{request.pass_type}_{request.filter_name}_{request.filename}.jpg"
        )
        plt.show()

        print(f"Request is done.\n")


if __name__ == "__main__":
    main()
