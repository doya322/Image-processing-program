import cv2
import numpy as np

def add(img1, img2):
    output_img = cv2.add(img1, img2)

    return output_img

def subtract(img1, img2):
    # BGR채널순서를 RGB채널로 변경
    RGB_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    RGB_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # RGB 채널 나누기
    R_img1, G_img1, B_img1 = cv2.split(RGB_img1)
    R_img2, G_img2, B_img2 = cv2.split(RGB_img2)
    # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
    R_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    G_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    B_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)

    # for문을 돌며 픽셀 빼기 연산 하기
    for h in range(RGB_img1.shape[0]):
        for w in range(RGB_img1.shape[1]):
            R_plus[h, w] = np.abs(np.int32(R_img1[h, w]) - np.int32(R_img2[h, w]))
            G_plus[h, w] = np.abs(np.int32(G_img1[h, w]) - np.int32(G_img2[h, w]))
            B_plus[h, w] = np.abs(np.int32(B_img1[h, w]) - np.int32(B_img2[h, w]))

    output_img[:, :, 0] = R_plus
    output_img[:, :, 1] = G_plus
    output_img[:, :, 2] = B_plus

    return output_img

def multiplication(img1, C):
    # BGR채널순서를 RGB채널로 변경
    RGB_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # RGB 채널 나누기
    R_img1, G_img1, B_img1 = cv2.split(RGB_img1)

    # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
    R_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    G_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    B_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)

    def saturation(value):  # saturation함수로 정의하기
        if (value > 255):
            value = 255;
        return value
        # for문을 돌며 픽셀 곱하기 연산 하기

    for h in range(RGB_img1.shape[0]):
        for w in range(RGB_img1.shape[1]):
            R_plus[h, w] = saturation(np.int32(R_img1[h, w]) * C)
            G_plus[h, w] = saturation(np.int32(G_img1[h, w]) * C)
            B_plus[h, w] = saturation(np.int32(B_img1[h, w]) * C)

    output_img[:, :, 0] = R_plus
    output_img[:, :, 1] = G_plus
    output_img[:, :, 2] = B_plus

    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    return output_img

def blending(img1, img2):
    # BGR채널순서를 RGB채널로 변경
    RGB_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    RGB_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # RGB 채널 나누기
    R_img1, G_img1, B_img1 = cv2.split(RGB_img1)
    R_img2, G_img2, B_img2 = cv2.split(RGB_img2)
    # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
    R_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    G_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    B_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)

    def saturation(value):  # saturation함수로 정의하기
        if (value > 255):
            value = 255;
        return value

    W = 0.4  # 가중치 설정
    # for문을 돌며 픽셀 블렌딩 연산 하기
    for h in range(RGB_img1.shape[0]):
        for w in range(RGB_img1.shape[1]):
            R_plus[h, w] = saturation(np.fabs(W * np.float32(R_img1[h, w]) + (1 - W) * np.float32(R_img2[h, w])))
            G_plus[h, w] = saturation(np.fabs(W * np.float32(G_img1[h, w]) + (1 - W) * np.float32(G_img2[h, w])))
            B_plus[h, w] = saturation(np.fabs(W * np.float32(B_img1[h, w]) + (1 - W) * np.float32(B_img2[h, w])))

    output_img[:, :, 0] = R_plus
    output_img[:, :, 1] = G_plus
    output_img[:, :, 2] = B_plus

    return output_img

def pixelAND(img1, img2):
    # BGR채널순서를 RGB채널로 변경
    RGB_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    RGB_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # RGB 채널 나누기
    R_img1, G_img1, B_img1 = cv2.split(RGB_img1)
    R_img2, G_img2, B_img2 = cv2.split(RGB_img2)
    # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
    R_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    G_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    B_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)

    def saturation(value):  # saturation함수로 정의하기
        if (value > 255):
            value = 255;
        return value

        # 영상 이진화 하기

    for h in range(RGB_img1.shape[0]):
        for w in range(RGB_img1.shape[1]):
            if (np.int32(R_img1[h, w]) > 180):
                R_img1[h, w] = G_img1[h, w] = B_img1[h, w] = 255
            else:
                R_img1[h, w] = G_img1[h, w] = B_img1[h, w] = 0
            if (np.int32(G_img2[h, w]) > 50):
                R_img2[h, w] = G_img2[h, w] = B_img2[h, w] = 255
            else:
                R_img2[h, w] = G_img2[h, w] = B_img2[h, w] = 0

                # for문을 돌며 픽셀 비트 AND 연산 하기
    for h in range(RGB_img1.shape[0]):
        for w in range(RGB_img1.shape[1]):
            R_plus[h, w] = saturation(np.int32(R_img1[h, w]) & np.int32(R_img2[h, w]))
            G_plus[h, w] = saturation(np.int32(G_img1[h, w]) & np.int32(G_img2[h, w]))
            B_plus[h, w] = saturation(np.int32(B_img1[h, w]) & np.int32(B_img2[h, w]))

    output_img[:, :, 0] = R_plus
    output_img[:, :, 1] = G_plus
    output_img[:, :, 2] = B_plus

    return output_img

def pixelOR(img1, img2):
    # BGR채널순서를 RGB채널로 변경
    RGB_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    RGB_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # RGB 채널 나누기
    R_img1, G_img1, B_img1 = cv2.split(RGB_img1)
    R_img2, G_img2, B_img2 = cv2.split(RGB_img2)
    # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
    R_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    G_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    B_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)

    def saturation(value):  # saturation함수로 정의하기
        if (value > 255):
            value = 255;
        return value

        # for문을 돌며 픽셀 비트 OR 연산 하기

    for h in range(RGB_img1.shape[0]):
        for w in range(RGB_img1.shape[1]):
            R_plus[h, w] = saturation(np.int32(R_img1[h, w]) | np.int32(R_img2[h, w]))
            G_plus[h, w] = saturation(np.int32(G_img1[h, w]) | np.int32(G_img2[h, w]))
            B_plus[h, w] = saturation(np.int32(B_img1[h, w]) | np.int32(B_img2[h, w]))

    output_img[:, :, 0] = R_plus
    output_img[:, :, 1] = G_plus
    output_img[:, :, 2] = B_plus

    return output_img

def pixelComplement(img1):
    # BGR채널순서를 RGB채널로 변경
    RGB_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # RGB 채널 나누기
    R_img1, G_img1, B_img1 = cv2.split(RGB_img1)
    # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
    R_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    G_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)
    B_plus = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]), dtype=np.ubyte)

    def saturation(value):  # saturation함수로 정의하기
        if (value > 255):
            value = 255;
        return value

        # for문을 돌며 픽셀 반전 연산 하기

    for h in range(RGB_img1.shape[0]):
        for w in range(RGB_img1.shape[1]):
            R_plus[h, w] = saturation(255 - np.int32(R_img1[h, w]))
            G_plus[h, w] = saturation(255 - np.int32(G_img1[h, w]))
            B_plus[h, w] = saturation(255 - np.int32(B_img1[h, w]))
    # 영상 다시 넣어주기
    output_img[:, :, 0] = R_plus
    output_img[:, :, 1] = G_plus
    output_img[:, :, 2] = B_plus

    return output_img

def globalThreshholding(img1):
    # BGR채널순서를 RGB채널로 변경
    RGB_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # 결과 영상을 담을 기억 장소 생성
    output_img = np.zeros((RGB_img1.shape[0], RGB_img1.shape[1]))

    # 영상 임계값 적용 하기
    for h in range(RGB_img1.shape[0]):
        for w in range(RGB_img1.shape[1]):
            if (np.int32(RGB_img1[h, w][0]) < 180):
                output_img[h, w] = 255
            else:
                output_img[h, w] = 0

    return output_img

def meanFiltering(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # 계수가 1로 구성된 3x3커널 만들기
    kernel = np.ones((3, 3), np.float32) / 9
    output_img = cv2.filter2D(gray_img, -1, kernel)

    return output_img

def medianFiltering(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # 5x5 중간값 커널 적용하기
    output_img = cv2.medianBlur(gray_img, 5)

    return output_img

def gaussianFiltering(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # 5x5 가우시안 커널 적용하기
    output_img = cv2.GaussianBlur(gray_img, (5, 5), 1)

    return output_img

def conservativeSmoothing(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    output_img = np.zeros((img1.shape[0], img1.shape[1]))
    center = 0
    current = 0
    min = 255;
    max = 0;
    ed = 1  # 3x3커널일 경우 1, 5x5 커널일 경우 2

    for h in range(ed, img1.shape[0] - ed, 1):
        for w in range(ed, img1.shape[1] - ed, 1):
            # 초기값 설정
            center = gray_img[h, w]
            min = gray_img[h - ed, w - ed]
            max = gray_img[h - ed, w - ed]
            # 최대, 최소 구하기
            for m in range(-ed, ed, 1):
                for n in range(-ed, ed, 1):
                    if (m == 0 and n == 0):
                        continue
                    else:
                        current = gray_img[h + m, w + n]
                    if (min > current):
                        min = current
                    if (max < current):
                        max = current
            if (center > min and center < max):
                output_img[h, w] = center
            elif (center > max):
                center = max
            elif (center < min):
                center = min
            output_img[h, w] = center

    return output_img

def unsharpFiltering(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # edge_img = np.zeros((img.shape[0],img.shape[1]))

    # 5x5커널 적용하기
    mean_img = cv2.blur(gray_img, (5, 5))

    edge_img = cv2.addWeighted(gray_img, 1.0, mean_img, -1.0, 0)
    output_img = cv2.addWeighted(gray_img, 1.0, edge_img, 3.0, 0)

    return output_img

def robertsCrossEdge(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 로버트 크로스 필터
    gx = np.array([[-1, 0], [0, 1]], dtype=int)
    gy = np.array([[0, -1], [1, 0]], dtype=int)

    # 로버트 크로스 컨벌루션
    x = cv2.filter2D(gray_img, -1, gx)
    y = cv2.filter2D(gray_img, -1, gy)

    # 절대값 취하기
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    output_img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return output_img

def sobelEdge(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Sobel operator
    x = cv2.Sobel(gray_img, -1, 1, 0)
    y = cv2.Sobel(gray_img, -1, 0, 1)

    # Turn uint8, image fusion
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    output_img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return output_img

def prewittEdge(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 프르윗 필터
    gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    # 프르윗 필터 컨벌루션
    x = cv2.filter2D(gray_img, -1, gx)
    y = cv2.filter2D(gray_img, -1, gy)
    # uint8 타입(0~255)로 변경하고 영상 합하기
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    output_img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return output_img

def cannyEdge(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 케니 에지 컨벌루션 연산하기
    output_img = cv2.Canny(gray_img, 100, 250)

    return output_img

def laplacianEdge(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 라플라시안 에지 컨벌루션 연산하기
    laplacian = cv2.Laplacian(gray_img, -1, 1)

    output_img = laplacian / laplacian.max()

    return output_img

def laplacianOfGaussianEdge(img1):
    # color영상을 gray영상으로 만들기
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray_img, (3, 3), 1)

    # 라플라시안 에지 컨벌루션 연산하기
    laplacian = cv2.Laplacian(blur, -1, 1)

    output_img = laplacian / laplacian.max()

    return output_img

def dilation(img1):
    kernel = np.ones((3, 3), np.uint8)
    output_img = cv2.dilate(img1, kernel, iterations=1)

    return output_img

def erosion(img1):
    kernel = np.ones((3, 3), np.uint8)
    output_img = cv2.erode(img1, kernel, iterations=1)

    return output_img

def opening(img1):
    kernel = np.ones((5, 5), np.uint8)
    output_img = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)

    return output_img

def closing(img1):
    kernel = np.ones((5, 5), np.uint8)
    output_img = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)

    return output_img

def rotateClockwise(img1):
    rows, cols = img1.shape[:2]
    # 회전점을 영상 모서리 -> 영상의 중심으로 변경
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 270, 1)
    output_img = cv2.warpAffine(img1, M, (cols * 1, rows * 1), flags=cv2.INTER_LINEAR)

    return output_img

def rotateCounterClockwise(img1):
    rows, cols = img1.shape[:2]
    # 회전점을 영상 모서리 -> 영상의 중심으로 변경
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
    output_img = cv2.warpAffine(img1, M, (cols * 1, rows * 1), flags=cv2.INTER_LINEAR)

    return output_img

def rotate180Deg(img1):
    rows, cols = img1.shape[:2]
    # 회전점을 영상 모서리 -> 영상의 중심으로 변경
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 180, 1)
    output_img = cv2.warpAffine(img1, M, (cols * 1, rows * 1), flags=cv2.INTER_LINEAR)

    return output_img

def horizontalFlip(img1):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    output_img = cv2.flip(img1, 1)

    return output_img

def verticalFlip(img1):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    output_img = cv2.flip(img1, 0)

    return output_img