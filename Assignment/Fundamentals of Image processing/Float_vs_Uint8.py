import cv2
import numpy as np


# data type을 명시하지 않으면 float
src1 = np.zeros((200,200))
src2 = np.ones((200,200))

src3 = np.zeros((200,200),dtype=np.uint8)
src4 = np.ones((200,200),dtype=np.uint8)
src5 = np.full((200,200),255,dtype=np.uint8)


print('<float>')
print(src1.shape, src2.shape)
print('src1 float')
print(src1)

print('<uint8>')
print(src3.shape, src4.shape,src5.shape)
print('src5 <uint8>')
print(src5)


cv2.imshow('src1',src1)
cv2.imshow('src2',src2)
cv2.imshow('src3',src3)
cv2.imshow('src4',src4)
cv2.imshow('src5',src5)


cv2.waitKey()
cv2.destroyAllWindows()

