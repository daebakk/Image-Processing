import cv2
import numpy as np

src = np.zeros((300,300,3),np.uint8)

# 2차원 이미지의 (0,0)위치에 BGR 채널 순으로 값을 넣는다.
src[0,0] = [1,2,3]
src[0,1] = [4,5,6]
src[1,0] = [7,8,9]


## 직접 해보는 것이 이해하기 더 쉽다!!!
print(src.shape)
#print(src[0,0,0])
#print(src[0,0,1])
#print(src[0,0])
#print(src[0])
print(src)

cv2.imshow('src',src)
cv2.waitKey()
cv2.destroyAllWindows()



