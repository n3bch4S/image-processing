import cv2 as cv
import numpy as np

from cv2.typing import MatLike


def showImage(image: MatLike) -> None:
    cv.imshow(winname="", mat=image)
    cv.waitKey(delay=0)
    cv.destroyAllWindows()


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


img: MatLike = cv.imread("flower.jpg")
showImage(image=img)
