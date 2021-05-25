import numpy as np
import matplotlib.pyplot as plt
import cv2


def my_calcHist_gray(mini_img):
    h, w = mini_img.shape[:2]  # (h,w) = img.shape
    hist = np.zeros((256,), dtype=np.int)
    #(256,) 과 (256,1)은 다르다
    # 전자는 1차원이고 후자는 2차원이다.
    # 따라서 hist는 256개의 원소를 가지는 1차원 배열이다.(0~255)

    for row in range(h):
        for col in range(w):
            intensity = mini_img[row, col] # intensity는 픽셀 값을 의미
            hist[intensity] += 1 # hist의 index는 intensity hist의 값은 해당 픽셀 값의 빈도수

    return hist


def my_hist_stretch(src,hist):

    (h,w) = src.shape
    dst = np.zeros((h,w),dtype=np.uint8)
    min = 256 # 최대값보다 1큰 값
    max = -1 # 최소값보다 1 작은 값

    # pixel intensity의 min값과 max값 구하기
    # hist의 index : pixel intensity,hist의 값 : pixel frequency

    for i in range(len(hist)):
        #
        if hist[i] != 0 and i < min:
            min = i
        if hist[i] != 0 and i > max:
            max = i

    # hist는 256개의 원소를 가지는 1차원 배열이다
    hist_stretch = np.zeros(hist.shape,dtype=np.int)

    for i in range(min,max+1):
        j = int((255 - 0) / (max - min) * (i - min) + 0)
        hist_stretch[j] = hist[i]


    for row in range(h):
        for col in range(w):
            dst[row,col] = (255 - 0) / (max - min) * (src[row,col] -min) + 0

    return dst, hist_stretch

if __name__ == '__main__':

    src = cv2.imread('../resources/fruits.jpg',cv2.IMREAD_GRAYSCALE)
    src_div = cv2.imread('fruits_div3.jpg',cv2.IMREAD_GRAYSCALE)

    #원본 이미지와 div 1/3 적용한 이미지의 histogram 구하기
    hist = my_calcHist_gray(src)
    hist_div = my_calcHist_gray(src_div)

    #div 1/3 적용한 이미지를 stretch 적용
    dst, hist_stretch = my_hist_stretch(src_div,hist_div)

    # div 1/3 이미지의 histogram

    binX = np.arange(len(hist_div))
    plt.bar(binX,hist_div,width=0.5,color='g')
    plt.title('divide 3 image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

    # div 1/3 이미지를 stretch 적용후 histogram
    binX = np.arange(len(hist_stretch))
    plt.bar(binX,hist_stretch,width=0.5,color='g')
    plt.title('stretching image applied by divide 3 image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()


    # 원본 이미지의 histogram
    binX = np.arange(len(hist))
    plt.bar(binX,hist,width=0.5,color='g')
    plt.title('original image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()


    # 이미지 출력
    cv2.imshow('original',src)
    cv2.imshow('div 1/3 image',src_div)
    cv2.imshow('stretched image',dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
