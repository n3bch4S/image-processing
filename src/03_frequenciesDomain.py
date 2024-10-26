import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from cv2.typing import MatLike
from numpy.typing import NDArray


class ResultImage:
    def __init__(
        self,
        filename: str,
        filter_name: str,
        pass_type: str,
        start_points: list[tuple[int, int]] | None = None,
        end_points: list[tuple[int, int]] | None = None,
    ):
        self.filename = filename
        self.filter_name = filter_name
        self.pass_type = pass_type
        self.start_points = start_points
        self.end_points = end_points


# region config
WANT_DISPLAY: bool = False

NOISY_TOM_JERRY: str = "Noisy_Tom_Jerry"
NOISY_WHIRLPOOL: str = "Noisy_whirlpool"
NOISY_GALAXY_3: str = "Noisy_galaxy3"

CUSTOM_FILTER: str = "CUSTOM_FILTER"
IDEAL_FILTER: str = "IDEAL_FILTER"
GAUSSIAN_FILTER: str = "GAUSSIAN_FILTER"

LOW_PASS: str = "LOW_PASS"
HIGH_PASS: str = "HIGH_PASS"

CUTOFFS: list[int] = [10, 50, 100]

RESULT_SET: list[ResultImage] = [
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
]

RESULT_SET_3: list[ResultImage] = [
    ResultImage(
        NOISY_TOM_JERRY,
        CUSTOM_FILTER,
        HIGH_PASS,
        start_points=[(147, 299), (162, 299)],
        end_points=[(158, 302), (173, 302)],
    ),
    ResultImage(
        NOISY_WHIRLPOOL,
        CUSTOM_FILTER,
        HIGH_PASS,
        start_points=[
            (128, 184),
            (128, 194),
            (138, 184),
            (138, 194),
            (118, 173),
            (117, 203),
            (147, 172),
            (147, 202),
        ],
        end_points=[
            (131, 187),
            (131, 197),
            (141, 187),
            (141, 197),
            (121, 179),
            (121, 208),
            (150, 178),
            (150, 208),
        ],
    ),
    ResultImage(
        NOISY_GALAXY_3,
        CUSTOM_FILTER,
        HIGH_PASS,
        start_points=[(288, 279), (288, 319)],
        end_points=[(292, 282), (292, 322)],
    ),
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


def custom_filter(
    shape: tuple[int, int],
    start_points: list[tuple[int, int]],
    end_points: list[tuple[int, int]],
    pass_type: str,
) -> NDArray[np.float32]:
    filter_mask: NDArray[np.float32] = np.zeros(shape, np.float32)

    for i in range(len(start_points)):
        x_start, y_start = start_points[i]
        x_end, y_end = end_points[i]

        filter_mask[x_start:x_end, y_start:y_end] = 1

    return filter_mask if pass_type == LOW_PASS else 1 - filter_mask


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


def classify_filter(
    request: ResultImage, shape: tuple[int, int], cutoffs: list[int], index: int
) -> NDArray[np.float32]:
    cutoff: int = CUTOFFS[index]
    filter: NDArray[np.float32] = np.zeros((1, 1), dtype=np.float32)

    if request.filter_name == IDEAL_FILTER:
        filter = ideal_filter(shape, cutoff, request.pass_type)
    elif request.filter_name == GAUSSIAN_FILTER:
        filter = gaussian_filter(shape, cutoff, request.pass_type)
    elif (
        request.filter_name == CUSTOM_FILTER
        and request.start_points is not None
        and request.end_points is not None
    ):
        filter = custom_filter(
            shape,
            request.start_points,
            request.end_points,
            request.pass_type,
        )
    else:
        filter = np.zeros(shape, np.float32) + 1
        print(f"No filter use at {index}")

    return filter


def main() -> None:
    # 1-2
    for request in RESULT_SET:
        print(f"Operating on request: {request.filename} with {request.filter_name}...")

        path: str = f"../assets/assignment_03/{request.filename}.jpg"
        image: NDArray[np.uint8] = cv.imread(path, cv.IMREAD_GRAYSCALE).astype(np.uint8)
        dft, spectrum = dft_and_spectrum(image)

        plt.figure(figsize=(10, 9))
        plot_image(image, f"Original {request.filename}", 4, 3, 1)
        plot_image(spectrum, f"Spectrum {request.filename}", 4, 3, 2)

        for i in range(len(CUTOFFS)):
            filter: NDArray[np.float32] = classify_filter(
                request, spectrum.shape, CUTOFFS, i
            )
            filtered_dft: NDArray[np.complex128] = (dft * filter).astype(np.complex128)
            spectrum = spectrum_of(filtered_dft)
            image = spatial_from(filtered_dft)

            cell_at: int = 3 * (i + 1)
            plot_image(
                scale_intensity(filter).astype(np.uint8),
                f"{request.pass_type} {request.filter_name} cutoff={CUTOFFS[i]}",
                *(4, 3, cell_at + 1),
            )
            plot_image(spectrum, f"Filtered Spectrum", *(4, 3, cell_at + 2))
            plot_image(image, f"After Filtered", *(4, 3, cell_at + 3))

        print(f"Saving result...")
        plt.savefig(
            f"../result/assignment_03/{request.pass_type}_{request.filter_name}_{request.filename}.jpg"
        )
        plt.show() if WANT_DISPLAY else 0

        print(f"Request is done.\n")

    # 3
    for request in RESULT_SET_3:
        print(f"Operating on request: {request.filename} with {request.filter_name}...")

        path: str = f"../assets/assignment_03/{request.filename}.jpg"
        image: NDArray[np.uint8] = cv.imread(path, cv.IMREAD_GRAYSCALE).astype(np.uint8)
        dft, spectrum = dft_and_spectrum(image)

        plt.figure(figsize=(10, 9))
        plot_image(image, f"Original {request.filename}", 2, 3, 1)
        plot_image(spectrum, f"Spectrum {request.filename}", 2, 3, 2)

        filter: NDArray[np.float32] = np.zeros((1, 1), dtype=np.float32)

        if (
            request.filter_name == CUSTOM_FILTER
            and request.start_points is not None
            and request.end_points is not None
        ):
            filter = custom_filter(
                spectrum.shape,
                request.start_points,
                request.end_points,
                request.pass_type,
            )
        else:
            filter = np.zeros_like(spectrum, np.float32) + 1
            print(f"No filter use for {request.filter_name}")

        filtered_dft: NDArray[np.complex128] = (dft * filter).astype(np.complex128)
        spectrum = spectrum_of(filtered_dft)
        image = spatial_from(filtered_dft)

        plot_image(
            scale_intensity(filter).astype(np.uint8),
            f"{request.pass_type} {request.filter_name}",
            *(2, 3, 4),
        )
        plot_image(spectrum, f"Filtered Spectrum", *(2, 3, 5))
        plot_image(image, f"After Filtered", *(2, 3, 6))

        print(f"Saving result...")
        plt.savefig(
            f"../result/assignment_03/{request.pass_type}_{request.filter_name}_{request.filename}.jpg"
        )
        plt.show() if WANT_DISPLAY else 0

        print(f"Request is done.\n")


if __name__ == "__main__":
    main()
