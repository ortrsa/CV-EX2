from ex2_utils import *
import matplotlib.pyplot as plt
import time
import cv2


def conv1Demo():
    print("\nconv1Demo:")
    array = np.array([1, 2, 3, 4, 5, 6, 7])
    k = np.array([-1, 1, 2])
    npc = np.convolve(array, k, 'full')
    my = conv1D(array, k)
    print("np.convolve : ", npc)
    print("my implementation:", my)
    print('np.convolve - my implementation shold be 0:', npc - my, '\n')


def conv2Demo():
    image = cv2.imread("spongebob.jpeg", 0)
    k = np.ones((3, 3)) /9
    cvc = cv2.filter2D(image, -1, k, borderType=cv2.BORDER_REPLICATE)
    my = conv2D(image, k)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()
    fig.suptitle('conv2Demo')
    ax1.set_title('cv2')
    ax1.imshow(cvc)
    ax2.set_title('my')
    ax2.imshow(my)
    plt.show()


def derivDemo():
    print("\nderivDemo:")
    image = cv2.imread("spongebob.jpeg", 0)
    directions, magnitude, x_der, y_der = convDerivative(image)
    print("directions:", directions)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.gray()
    fig.suptitle('derivDemo')
    ax1.set_title('x_der')
    ax1.imshow(x_der)
    ax2.set_title('y_der')
    ax2.imshow(y_der)
    ax3.set_title('magnitude')
    ax3.imshow(magnitude)
    plt.show()


def blurDemo():

    image = cv2.imread("spongebob.jpeg", 0)
    myblur = blurImage1(image,3)
    cvblur = blurImage2(image,3)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()
    fig.suptitle('conv2Demo')
    ax1.set_title('myblur')
    ax1.imshow(myblur)
    ax2.set_title('cvblur')
    ax2.imshow(cvblur)
    plt.show()


def edgeDemo():
    image = cv2.imread("batterfly.jpeg", 0)
    sob_my, sob_cv = edgeDetectionSobel(image)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()
    fig.suptitle('sobel')
    ax1.set_title('my sobel')
    ax1.imshow(sob_my)
    ax2.set_title('cv2 sobel')
    ax2.imshow(sob_cv)
    plt.show()

    image = cv2.imread("camera.jpeg", 0)
    LoG = edgeDetectionZeroCrossingLOG(image)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()
    fig.suptitle('LoG')
    ax1.set_title('LoG')
    ax1.imshow(LoG)
    ax2.set_title('image before LoG')
    ax2.imshow(image)
    plt.show()

    image = cv2.imread("daryan.jpeg", 0)
    canny_cv, canny_my = edgeDetectionCanny(image, 0.75, 0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()
    fig.suptitle('canny')
    ax1.set_title('canny_cv')
    ax1.imshow(canny_cv)
    ax2.set_title('canny_my')
    ax2.imshow(canny_my)
    plt.show()


def houghDemo():
    print("\nhoughDemo:")
    image = cv2.imread('circle.jpeg', 0)
    image1 = cv2.imread('circle.jpeg', 1)
    c = houghCircle(image, 42, 43)
    if c is not None:
        circles = c
        for i in circles:
            center = (i[1], i[0])
            radius = i[2]
            cv2.circle(image1, center, radius, (124, 252, 0), 3)
    plt.title('houghDemo(R = 42 - 43)')
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.show()


def main():
    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo()
    edgeDemo()
    houghDemo()


if __name__ == '__main__':
    main()
