import cv2
import numpy as np


"""
YUV란?

빛의 밝기를 나타내는 휘도(Y)와 색상신호 2개(U, V)로 표현하는 방식이다.

The Y′UV model defines a color space in terms of one luma component (Y′) and 
two chrominance components, called U (blue projection) and V (red projection) respectively.

Y = 0.299R + 0.587G + 0.114B
"""

src = cv2.imread('../resources/Lena.png')

(h,w,c) = src.shape
yuv = cv2.cvtColor(src,cv2.COLOR_BGR2YUV)
my_y = np.zeros((h,w))
my_y = (src[:,:,0] * 0.114) + (src[:,:,1] * 0.587) + (src[:,:,2] * 0.299)
my_y = np.round(my_y).astype(np.uint8)


print(yuv[0:5,0:5,0])
print(my_y[0:5,0:5])

cv2.imshow('original',src)
cv2.imshow('cvtColor',yuv[:,:,0]) # yuv의 첫번째 채널은 y
cv2.imshow('my_y',my_y)

cv2.waitKey()
cv2.destroyAllWindows()
