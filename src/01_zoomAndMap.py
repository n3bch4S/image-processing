from typing import Callable
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
    return cv.resize(
        src=image,
        dsize=None,
        fx=zoomFactor,
        fy=zoomFactor,
        interpolation=cv.INTER_NEAREST,
    )


def bilinearZoom(zoomFactor: float, image: MatLike) -> MatLike:
    return cv.resize(
        src=image,
        dsize=None,
        fx=zoomFactor,
        fy=zoomFactor,
        interpolation=cv.INTER_LINEAR,
    )


# TODO: no nested loop
def enhanceGrayScale(
    image: MatLike, function: Callable[[np.uint8], np.uint8]
) -> MatLike:
    width: int = image.shape[1]
    height: int = image.shape[0]
    newImage: NDArray = np.zeros(shape=(height, width), dtype=image.dtype)

    for m in range(height):
        for n in range(width):
            newImage[m, n] = function(image[m, n])
    return newImage


# TODO: can all variable be np.dtype(number)
def makeGivenFunction(l: int):
    # range by L
    lBy3: int = l // 3
    doubleLBy3: int = 2 * l // 3
    fifthLBy6: int = 5 * l // 6

    # linear constant
    m: float = -2
    c: float = 3 * l / 2

    def givenFunction(r: np.uint8) -> np.uint8:
        if r <= lBy3 or r >= doubleLBy3:
            return np.uint8(fifthLBy6)
        else:
            return np.uint8(m * r.astype(int) + c)

    return givenFunction


# TODO: just make transformation function not MatLike transformer
def makePowerLawTransformation(c: float, gamma: float):
    def powerLawTransformation(image: MatLike) -> MatLike:
        width: int = image.shape[1]
        height: int = image.shape[0]
        newImage: NDArray = np.zeros(shape=(height, width), dtype=image.dtype)

        for m in range(height):
            for n in range(width):
                normalizeIntensity: int = image[m, n] // 8
                newImage[m, n] = np.uint8(min(c * normalizeIntensity**gamma * 255, 255))
        return newImage

    return powerLawTransformation


fileDirectory: list[str] = [
    "../assets/assgn_01/flower.jpg"  # index 0
    "../assets/assgn_01/fractal.jpg"  # index 1
    "../assets/assgn_01/galaxy3.jpeg"  # index 2
    "../assets/assgn_01/spellbound_mini.jpg"  # index 3
    "../assets/assgn_01/superpets_mini.jpg"  # index 4
    "../assets/assgn_01/traffic.jpg"  # index 5
]
# TODO: can turn all case to dict...
# replicate x3 flower
image: MatLike = readImageGray(filename="../assets/assgn_01/flower.jpg")
zoomImage: MatLike = replicationZoom(zoomFactor=3, image=image)
filename: str = "replicateX3Flower"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# replicate x3 fractal
image: MatLike = readImageGray(filename="../assets/assgn_01/fractal.jpg")
zoomImage: MatLike = replicationZoom(zoomFactor=3, image=image)
filename: str = "replicateX3Fractal"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# replicate x3 traffic
image: MatLike = readImageGray(filename="../assets/assgn_01/traffic.jpg")
zoomImage: MatLike = replicationZoom(zoomFactor=3, image=image)
filename: str = "replicateX3Traffic"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# replicate x1/3 flower
image: MatLike = readImageGray(filename="../assets/assgn_01/flower.jpg")
zoomImage: MatLike = replicationZoom(zoomFactor=1 / 3, image=image)
filename: str = "replicateX1_3Flower"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# replicate x1/3 fractal
image: MatLike = readImageGray(filename="../assets/assgn_01/fractal.jpg")
zoomImage: MatLike = replicationZoom(zoomFactor=1 / 3, image=image)
filename: str = "replicateX1_3Fractal"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# replicate x1/3 traffic
image: MatLike = readImageGray(filename="../assets/assgn_01/traffic.jpg")
zoomImage: MatLike = replicationZoom(zoomFactor=1 / 3, image=image)
filename: str = "replicateX1_3Traffic"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# bilinear x3 flower
image: MatLike = readImageGray(filename="../assets/assgn_01/flower.jpg")
zoomImage: MatLike = bilinearZoom(zoomFactor=3, image=image)
filename: str = "bilinearX3Flower"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# bilinear x3 fractal
image: MatLike = readImageGray(filename="../assets/assgn_01/fractal.jpg")
zoomImage: MatLike = bilinearZoom(zoomFactor=3, image=image)
filename: str = "bilinearX3Fractal"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# bilinear x3 traffic
image: MatLike = readImageGray(filename="../assets/assgn_01/traffic.jpg")
zoomImage: MatLike = bilinearZoom(zoomFactor=3, image=image)
filename: str = "bilinearX3Traffic"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# bilinear x1/3 flower
image: MatLike = readImageGray(filename="../assets/assgn_01/flower.jpg")
zoomImage: MatLike = bilinearZoom(zoomFactor=1 / 3, image=image)
filename: str = "bilinearX1_3Flower"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# bilinear x1/3 fractal
image: MatLike = readImageGray(filename="../assets/assgn_01/fractal.jpg")
zoomImage: MatLike = bilinearZoom(zoomFactor=1 / 3, image=image)
filename: str = "bilinearX1_3Fractal"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)

# bilinear x1/3 traffic
image: MatLike = readImageGray(filename="../assets/assgn_01/traffic.jpg")
zoomImage: MatLike = bilinearZoom(zoomFactor=1 / 3, image=image)
filename: str = "bilinearX1_3Traffic"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=zoomImage)


givenFunction: Callable[[np.uint8], np.uint8] = makeGivenFunction(256)
# enhance grayscale with givenFunction superpets_mini
image: MatLike = readImageGray(filename="../assets/assgn_01/superpets_mini.jpg")
enhanceImage: MatLike = enhanceGrayScale(image=image, function=givenFunction)
filename: str = "givenFunction_superpets_mini"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

# enhance grayscale with givenFunction traffic
image: MatLike = readImageGray(filename="../assets/assgn_01/traffic.jpg")
enhanceImage: MatLike = enhanceGrayScale(image=image, function=givenFunction)
filename: str = "givenFunction_traffic"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

# enhance grayscale with givenFunction spellbound_mini
image: MatLike = readImageGray(filename="../assets/assgn_01/spellbound_mini.jpg")
enhanceImage: MatLike = enhanceGrayScale(image=image, function=givenFunction)
filename: str = "givenFunction_spellbound_mini"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

powerLaw = makePowerLawTransformation(c=0.5, gamma=0.4)
# powerLaw 0_5 0_4 superpets_mini
image: MatLike = readImageGray(filename="../assets/assgn_01/superpets_mini.jpg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 0_5 0_4 superpets_mini"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

# powerLaw 0_5 0_4 galaxy3
image: MatLike = readImageGray(filename="../assets/assgn_01/galaxy3.jpeg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 0_5 0_4 galaxy3"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

# powerLaw 0_5 0_4 traffic
image: MatLike = readImageGray(filename="../assets/assgn_01/traffic.jpg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 0_5 0_4 traffic"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

powerLaw = makePowerLawTransformation(c=0.5, gamma=2.5)
# powerLaw 0_5 2_5 superpets_mini
image: MatLike = readImageGray(filename="../assets/assgn_01/superpets_mini.jpg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 0_5 2_5 superpets_mini"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

# powerLaw 0_5 2_5 galaxy3
image: MatLike = readImageGray(filename="../assets/assgn_01/galaxy3.jpeg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 0_5 2_5 galaxy3"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

# powerLaw 0_5 2_5 traffic
image: MatLike = readImageGray(filename="../assets/assgn_01/traffic.jpg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 0_5 2_5 traffic"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

powerLaw = makePowerLawTransformation(c=1.2, gamma=0.4)
# powerLaw 1_2 0_4 superpets_mini
image: MatLike = readImageGray(filename="../assets/assgn_01/superpets_mini.jpg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 1_2 0_4 superpets_mini"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

# powerLaw 1_2 0_4 galaxy3
image: MatLike = readImageGray(filename="../assets/assgn_01/galaxy3.jpeg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 1_2 0_4 galaxy3"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

# powerLaw 1_2 0_4 traffic
image: MatLike = readImageGray(filename="../assets/assgn_01/traffic.jpg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 1_2 0_4 traffic"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

powerLaw = makePowerLawTransformation(c=1.2, gamma=2.5)
# powerLaw 1_2 2_5 superpets_mini
image: MatLike = readImageGray(filename="../assets/assgn_01/superpets_mini.jpg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 1_2 2_5 superpets_mini"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

# powerLaw 1_2 2_5 galaxy3
image: MatLike = readImageGray(filename="../assets/assgn_01/galaxy3.jpeg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 1_2 2_5 galaxy3"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)

# powerLaw 1_2 2_5 traffic
image: MatLike = readImageGray(filename="../assets/assgn_01/traffic.jpg")
enhanceImage: MatLike = powerLaw(image=image)
filename: str = "powerLaw 1_2 2_5 traffic"
cv.imwrite(filename=f"../result/assgn_01/{filename}.jpg", img=enhanceImage)
