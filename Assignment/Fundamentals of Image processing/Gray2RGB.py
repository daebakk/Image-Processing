import cv2
import numpy as np

"""

OPENCV는 RGB 순서가 아닌 GBR 순서

Y = 0.299R + 0.587G + 0.114B

Gray to RGB[A]:R←Y,G←Y,B←Y,A←max(ChannelRange)

"""
def my_gray2rgb(src):
    '''
    :param src:컬러 이미지
    :return dst1, dst2, dst3:흑백 이미지
    '''

    (h,w,c) = src.shape

    dst = np.zeros((h,w,c))
    y = np.zeros((h,w))
    # Y <- 0.299⋅R+0.587⋅G+0.114⋅B
    y = (src[:,:,0] * 0.114) + (src[:,:,1] * 0.587) + (src[:,:,2] * 0.299)
    y = np.round(y).astype(np.uint8)
    print(y)
    #dst[:,:,0] = y
    #dst[:,:,1] = y
    #dst[:,:,2] = y
    return dst


#아래의 이미지 3개 다 해보기
src = cv2.imread('../resources/fruits.jpg')
#src = cv2.imread('Lena.png')
#src = cv2.imread('Penguins.png')

dst = my_gray2rgb(src)

#cv2.imshow('original', src)
#cv2.imshow('rgb', dst)

#cv2.waitKey()
#cv2.destroyAllWindows()
