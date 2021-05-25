import cv2
import numpy as np

def my_nearest_neighbor(src, scale):

    (h,w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst), np.uint8)
    for row in range(h_dst):
        for col in range(w_dst):
            r = min(int(row / scale + 0.5), h-1)
            c = min(int(col / scale + 0.5), w-1)
            dst[row,col] = src[r,c]

    return dst



if __name__ == '__main__':
    src = cv2.imread('Lena.png',cv2.IMREAD_GRAYSCALE)


    # scale은 사용자 정의 (크기를 줄이는 방향으로)
    scale = 1/2

    dst_nearest_div2 = cv2.resize(src, dsize=(0,0), fx=scale,fy=scale,
                                  interpolation=cv2.INTER_NEAREST)
    dst_my_nearest_div2 = my_nearest_neighbor(src,scale)

    cv2.imshow('original',src)
    cv2.imshow('dst_nearest 1/3',dst_nearest_div2)
    cv2.imshow('my dst_nearest 1/3', dst_my_nearest_div2)

    # 크기를 늘리는 방향으로
    scale = 1 / scale
    dst_nearest = cv2.resize(dst_nearest_div2,dsize=(0,0), fx=scale,
                             fy=scale, interpolation=cv2.INTER_NEAREST)
    dst_nearest = my_nearest_neighbor(dst_my_nearest_div2,scale)

    cv2.imshow('dst_nearest',dst_nearest)
    cv2.imshow('my dst_nearest',dst_nearest)

    cv2.waitKey()
    cv2.destroyAllWindows()

