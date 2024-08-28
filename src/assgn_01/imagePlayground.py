import cv2 as cv
import numpy as np

from cv2.typing import MatLike
from numpy.typing import NDArray


def readImageGray(filename: str) -> MatLike:
    return cv.imread(filename=filename, flags=cv.IMREAD_GRAYSCALE)


def showImage(image: MatLike) -> None:
    cv.imshow(winname="Thada", mat=image)
    cv.waitKey(delay=0)
    cv.destroyAllWindows()


def replicationZoom(zoomFactor: float, image: MatLike) -> MatLike:
    newWidth: int = int(image.shape[1] * zoomFactor)
    newHeight: int = int(image.shape[0] * zoomFactor)
    newImage: NDArray = np.zeros(shape=(newHeight, newWidth), dtype=image.dtype)

    # TODO: maybe can use map function or any numpy dStruct(ogrid) for better performance
    # but nested loop is not that bad
    for m in range(newHeight):
        for n in range(newWidth):
            oldM: int = int(m / zoomFactor)
            oldN: int = int(n / zoomFactor)
            newImage[m, n] = image[oldM, oldN]
    return newImage


def bilinearZoom(zoomFactor: float, image: MatLike) -> MatLike:
    newWidth: int = int(image.shape[1] * zoomFactor)
    newHeight: int = int(image.shape[0] * zoomFactor)

    return cv.resize(
        src=image, dsize=(newWidth, newHeight), interpolation=cv.INTER_LINEAR
    )


def makeGivenFunction(l: int):
    # range by L
    lBy3: int = l // 3
    doubleLBy3: int = 2 * l // 3
    fifthLBy6: int = 5 * l // 6

    # linear constant
    m: float = -2
    c: float = 3 * l / 2

    def givenFunction(r: int) -> int:
        if r <= lBy3 or r >= doubleLBy3:
            return fifthLBy6
        else:
            return int(m * r + c)

    return givenFunction


def makePowerLawTransformation(c: float, gamma: float):
    def powerLawTransformation(image: MatLike) -> MatLike:
        return c * np.power(image, gamma)

    return powerLawTransformation


image: MatLike = readImageGray(filename="../../assets/flower.jpg")
zoomImage: MatLike = replicationZoom(zoomFactor=3, image=image)
showImage(image=zoomImage)
