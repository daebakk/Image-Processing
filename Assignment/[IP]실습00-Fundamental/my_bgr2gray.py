import cv2
import numpy as np

"""

Y = 0.299R + 0.587G + 0.114B

RGB[A] to Gray: Y <- 0.299⋅R+0.587⋅G+0.114⋅B

Gray to RGB[A]:R←Y,G←Y,B←Y,A←max(ChannelRange)

"""
def my_bgr2gray(src):
    '''
    :param src:컬러 이미지
    :return dst1, dst2, dst3:흑백 이미지
    '''

    (h,w,c) = src.shape

    #cvtColor() 함수 이용
    dst1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    #########################
    # TODO                  #
    # dst2, dst3 채우기     #
    #########################
    #dst2는 B, G, R채널 각각 1/3씩 사용
    dst2 = (src[:,:,0] * 1 / 3) + (src[:,:,1] * 1 / 3) + (src[:,:,2] * 1 / 3)

    #dst3은 B, G, R채널 중 하나의 채널만 사용(B,G,R중 원하는거 아무거나)
    dst3 = (src[:,:,0] * 1) + (src[:,:,1] * 0) + (src[:,:,2] * 0)

    y = np.zeros((h,w))
    # Y <- 0.299⋅R+0.587⋅G+0.114⋅B
    y = (src[:,:,0] * 0.114) + (src[:,:,1] * 0.587) + (src[:,:,2] * 0.299)
    y = np.round(y).astype(np.uint8)
    dst4 = y
    #dst2 반올림 np.round를 사용해도 무관
    dst2 = (dst2+0.5).astype(np.uint8)

    return dst1, dst2, dst3,dst4


#아래의 이미지 3개 다 해보기
src = cv2.imread('fruits.jpg')
#src = cv2.imread('Lena.png')
#src = cv2.imread('Penguins.png')

dst1, dst2, dst3,dst4 = my_bgr2gray(src)

cv2.imshow('original', src)
cv2.imshow('gray(cvtColor)', dst1)
cv2.imshow('gray(1/3)', dst2)
cv2.imshow('gray(one channel)', dst3)
cv2.imshow('gray(Y)',dst4)


cv2.waitKey()
cv2.destroyAllWindows()
