import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_calcHist(src):
    ###############################
    # TODO                        #
    # my_calcHist완성             #
    # src : input image           #
    # hist : src의 히스토그램      #
    ###############################

    # src의 크기를 나타태는 값 ex) 여기서는 480 x 512
    # 여기서는 h = 480, w = 512
    (h,w) = src.shape
    # 각 픽섹값의 빈도수를 저장하는 변수
    # hist의 index는 픽셀값 자체를 의미하고 hist의 값은 해당 픽셀값의 빈도수를 의미
    hist = np.zeros((256,), dtype=np.int)
    for row in range(h):
        for col in range(w):
            intensity = src[row, col]  # intensity는 픽셀 값을 의미
            hist[intensity] += 1  # hist의 index는 intensity hist의 값은 해당 픽셀 값의 빈도수

    return hist

def my_normalize_hist(hist, pixel_num):
    ########################################################
    # TODO                                                 #
    # my_normalize_hist완성                                #
    # hist : 히스토그램                                     #
    # pixel_num : image의 전체 픽셀 수                      #
    # normalized_hist : 히스토그램값을 총 픽셀수로 나눔      #
    ########################################################

    normalized_hist = hist / pixel_num
    return normalized_hist


def my_PDF2CDF(pdf):
    ########################################################
    # TODO                                                 #
    # my_PDF2CDF완성                                       #
    # pdf : normalized_hist                                #
    # cdf : pdf의 누적                                     #
    ########################################################

    cdf = np.zeros(pdf.shape)
    # cdf 초기화
    cdf[0] = pdf[0]
    # cdf_n = pdf_n + cdf_(n-1)
    for i in range(1,len(pdf)):
        cdf[i] = pdf[i] + cdf[i-1]
    return cdf


def my_denormalize(normalized, gray_level):
    ########################################################
    # TODO                                                 #
    # my_denormalize완성                                   #
    # normalized : 누적된pdf값(cdf)                        #
    # gray_level : max_gray_level                          #
    # denormalized : normalized와 gray_level을 곱함        #
    ########################################################

    denormalized = normalized * gray_level
    return denormalized


def my_calcHist_equalization(denormalized, hist):
    ###################################################################
    # TODO                                                            #
    # my_calcHist_equalization완성                                    #
    # denormalized : output gray_level(정수값으로 변경된 gray_level)   #
    # hist : 히스토그램                                                #
    # hist_equal : equalization된 히스토그램                           #
    ####################################################################

    hist_equal = np.zeros(hist.shape)
    for i in range(len(hist_equal)):
        hist_equal[i] = np.sum(hist[denormalized == i])

    return hist_equal


def my_equal_img(src, output_gray_level):
    ###################################################################
    # TODO                                                            #
    # my_equal_img완성                                                #
    # src : input image                                               #
    # output_gray_level : denormalized(정수값으로 변경된 gray_level)   #
    # dst : equalization된 결과 이미지                                 #
    ####################################################################

    (h,w) = src.shape
    dst = np.zeros((h,w),dtype=np.uint8)

    for row in range(h):
        for col in range(w):
            dst[row,col] = output_gray_level[src[row,col]]
    return dst

#input_image의  equalization된 histogram & image 를 return
def my_hist_equal(src):
    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram)
    denormalized_output = my_denormalize(normalized_output, max_gray_level)
    output_gray_level = denormalized_output.astype(int)
    hist_equal = my_calcHist_equalization(output_gray_level, histogram)

    # show mapping function
    ###################################################################
    # TODO                                                            #
    # plt.plot(???,???)완성                                           #
    # plt.plot(y축, x축)                                               #
    ###################################################################

    plt.plot(np.array(range(256)),output_gray_level)
    plt.title('mapping function')
    plt.xlabel('input intensity')
    plt.ylabel('output intensity')
    plt.show()

    ### dst : equalization 결과 image
    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal

if __name__ == '__main__':

    src = cv2.imread('fruits_div3.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(src)
    dst, hist_equal = my_hist_equal(src)
    #hist_equal = my_hist_equal(src)

    plt.figure(figsize=(8, 5))
    cv2.imshow('original', src)
    binX = np.arange(len(hist))
    plt.title('my histogram')
    plt.bar(binX, hist, width=0.5, color='g')
    plt.show()


    plt.figure(figsize=(8, 5))
    cv2.imshow('equalizetion after image', dst)
    binX = np.arange(len(hist_equal))
    plt.title('my histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()


    cv2.waitKey()
    cv2.destroyAllWindows()

