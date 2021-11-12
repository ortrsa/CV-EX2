import cv2
import numpy as np



def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """

    kernel1 = kernel1[::-1]
    ksize = len(kernel1)
    arrsize = len(inSignal)
    # make array that will fit conv1
    ans = np.zeros(ksize + arrsize - 1)
    # make array padded with '0's
    arr = np.zeros(arrsize + 2 * (ksize - 1))
    arr[ksize - 1:ksize + arrsize - 1] = inSignal

    for i in range(ksize + arrsize - 1):
        ans[i] = ((kernel1 * arr[i:ksize + i]).sum())
    return ans


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    inImage = inImage.astype(np.uint8)

    # Flip Kernel
    kernel2 = np.flip(kernel2)

    # Kernel ,Image and Padding shapes
    xksize = kernel2.shape[0]
    yksize = kernel2.shape[1]
    ximsize = inImage.shape[0]
    yimsize = inImage.shape[1]

    output = np.zeros((ximsize, yimsize))
    padding_x = int((xksize / 2) + 1)
    padding_y = int((yksize / 2) + 1)
    impad = cv2.copyMakeBorder(
        inImage,
        top=padding_y,
        bottom=padding_y,
        left=padding_x,
        right=padding_x,
        borderType=cv2.BORDER_REPLICATE
    )

    # Iterate through image
    for y in range(inImage.shape[1]):
        if y > inImage.shape[1]:
            break
        for x in range(inImage.shape[0]):
            if x > inImage.shape[0]:
                break

            output[x, y] = (impad[x: x + xksize, y: y + yksize] * kernel2).sum()

    output[output < 0] = 0

    return output.round()


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """
    if inImage.max() > 2:
        inImage = inImage / 255
    ximsize = inImage.shape[0]
    yimsize = inImage.shape[1]

    x_d = np.array([[1, 0, -1]])
    x_der = cv2.filter2D(inImage, -1, x_d, borderType=cv2.BORDER_REPLICATE)

    y_d = np.flip(np.array([[1, 0, -1]]).transpose())
    y_der = cv2.filter2D(inImage, -1, y_d, borderType=cv2.BORDER_REPLICATE)

    magnitude = np.sqrt(((x_der ** 2) + (y_der ** 2))).astype(np.uint8)

    directions = np.arctan2(y_der, x_der)
    return directions, magnitude, x_der, y_der


def blurImage1(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    ax = np.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
    s = 0.3 * (((kernel_size - 1)/2) - 1) + 0.8
    x, y = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(s))
    return conv2D(in_image, k)


def blurImage2(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    k = cv2.getGaussianKernel(kernel_size, -1)
    img = cv2.filter2D(in_image, -1, k, borderType=cv2.BORDER_REPLICATE)
    return img


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
     Detects edges using the Sobel method
     :param img: Input image
     :param thresh: The minimum threshold for the edge response
     :return: opencv solution, my implementation
     """
    if img.max() > 2:
        img = img / 255
    x_mat = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])

    y_mat = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])

    G_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    G_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel1 = ((G_x ** 2) + (G_y ** 2)) ** 0.5

    G_x = cv2.filter2D(img, -1, x_mat, borderType=cv2.BORDER_REPLICATE)
    G_y = cv2.filter2D(img, -1, y_mat, borderType=cv2.BORDER_REPLICATE)
    sobel2 = ((G_x ** 2) + (G_y ** 2)) ** 0.5

    ans1 = (np.where(sobel1 > thresh, 1, 0))
    ans2 = (np.where(sobel2 > thresh, 1, 0))
    return ans1, ans2


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    if img.max() > 2:
        img = img / 255
    Lap_K = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
    lap = cv2.filter2D(img, -1, Lap_K, borderType=cv2.BORDER_REPLICATE).round(3)
    lap[np.abs(lap) < 0.005] = 0
    ans = np.zeros(img.shape)
    # iterate throw the image
    for i in range(1, lap.shape[0] - 1):
        for j in range(1, lap.shape[1] - 1):
            # split to up-left neighbour and down-right neighbour
            up_l_neighbour = np.array([lap[i, j - 1], lap[i - 1, j - 1], lap[i - 1, j], lap[i - 1, j + 1]])
            down_r_neighbour = np.array([lap[i, j + 1], lap[i + 1, j + 1], lap[i + 1, j], lap[i + 1, j - 1]])
            # if the pixel is 0 look for [-,0,+] or [+,0,-]
            if lap[i, j] == 0:
                for u, d in zip(up_l_neighbour, down_r_neighbour):
                    if u > 0:
                        ans[i, j] = int(d < 0)
                    elif u < 0:
                        ans[i, j] = int(d > 0)
            #  if the pixel is negative check if one of his nei is positive
            if lap[i, j] < 0:
                if ((up_l_neighbour > 0).sum() + (down_r_neighbour > 0).sum()) > 0:
                    ans[i, j] = 1
            #  if the pixel is positive check if one of his nei is negative
            if lap[i, j] > 0:
                if ((up_l_neighbour < 0).sum() + (down_r_neighbour < 0).sum()) > 0:
                    ans[i, j] = 1
    return ans


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: :return: Edge matrix
    """

    return edgeDetectionZeroCrossingSimple(cv2.GaussianBlur(img, (5, 5), 0))


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    opencv = cv2.Canny(img, img.shape[0], img.shape[1])

    if img.max() > 2:
        img = img / 255
    # get answer from sobel and normalize
    G_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    G_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    img = ((G_x ** 2) + (G_y ** 2)) ** 0.5

    # get directions from 0 to 180 and Quantize to 4 values
    directions = (np.arctan2(G_x, G_y) * 180) / np.pi
    directions %= 180
    directions[(directions >= 0) & (directions < 22.5)] = 0
    directions[(directions >= 157.5) & (directions < 180)] = 0
    directions[(directions >= 22.5) & (directions < 67.5)] = 45
    directions[(directions >= 67.5) & (directions < 112.5)] = 90
    directions[(directions >= 112.5) & (directions < 157.5)] = 135

    # NMS
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            nei_l = 0
            nei_r = 0
            if directions[i, j] == 0:
                nei_l = img[i - 1, j]
                nei_r = img[i + 1, j]
            elif directions[i, j] == 45:
                nei_l = img[i - 1, j - 1]
                nei_r = img[i + 1, j + 1]
            elif directions[i, j] == 90:
                nei_l = img[i, j - 1]
                nei_r = img[i, j + 1]
            elif directions[i, j] == 135:
                nei_l = img[i - 1, j + 1]
                nei_r = img[i + 1, j - 1]
            if (img[i, j] < nei_l) | (img[i, j] < nei_r):
                img[i, j] = 0
    # value of 1 to pixels that are greater then T1
    # and 0 to pixels that smaller then T2
    img[img >= thrs_1] = 1
    img[img <= thrs_2] = 0
    # if the pixel value is greater then T2 and smaller then T1 mark by value 3.
    img[(thrs_2 < img) & (img < thrs_1)] = 3

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == 3:
                #  if the pixel value is 3 check if one of his neighbors is 1
                if 1 in img[i - 1:i + 2, j - 1:j + 2]:
                    img[i, j] = 1
                else:
                    img[i, j] = 0
    # if there is any left marks change to 0. (usually at borders)
    img[img == 3] = 0
    img = (img * 255).astype(np.uint8)
    return opencv, img


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension :param I: Input image
    :param img: image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """

    rows = img.shape[0]
    cols = img.shape[1]

    img, _ = edgeDetectionCanny(img, 0.75, 0.2)
    radius = range(min_radius, max_radius)
    circles = []
    threshold = 160
    for r in radius:
        print('radius: ', r)
        acc = np.zeros(img.shape)
        # Make accumulator
        for i in range(rows):
            for j in range(cols):
                if img[i, j] == 255:
                    for angle in range(360):
                        b = j - round(np.sin(angle * np.pi / 180) * r)
                        a = i - round(np.cos(angle * np.pi / 180) * r)
                        if 0 <= a < rows and 0 <= b < cols:
                            acc[a, b] += 1

        if acc.max() > threshold:
            acc[acc < threshold] = 0
            # find the circles for this radius
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if acc[i, j] >= threshold:
                        avg_sum = acc[i - 1:i + 2, j - 1:j + 2].sum() / 9
                        if avg_sum >= threshold / 9:
                            # checking that the distance from every circle to the current circle
                            # is more than the radius
                            if all((i - xc) ** 2 + (j - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
                                circles.append((i, j, r))
                                acc[i - r:i + r, j - r:j + r] = 0
    return circles
