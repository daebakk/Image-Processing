import cv2
import numpy as np

def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))
    print('original shape : {}'.format(src.shape))
    print('dst shape : {}'.format(dst.shape))
    # bilinear interpolation 적용
    for row in range(h_dst):
        for col in range(w_dst):
            # 참고로 꼭 한줄로 구현해야 하는건 아닙니다 여러줄로 하셔도 상관없습니다.(저도 엄청길게 구현했습니다.)
            # 스케일링 되기 전 original 좌표
            y = row / scale
            x = col / scale

            # bilinear interpolation
            # 1.(y,x)를 기준으로 좌측위,좌측아래,우측아래,우측위 좌표를 구함.
            # 2. bilinear interplation 식에 따라 해당 row,col좌표에 값을 대입
            y_up = int(y) # 버림
            y_down = min(int(y+1),h-1) # 반올림 단 src의 최대 좌표값보다는 같거나 작게
            x_left = int(x) # 버림
            x_right = min(int(x+1),w-1) # 반올림 단 src의 최대 좌표값보다는 같거나 작게
            print('dst up down left right : {} {} {} {}'.format(y_up,y_down,x_left,x_right))
            t = y - y_up
            s = x - x_left

            # Index를 초과하는 값에 대해서는 값 참조 안함
            if y_up < 0 or y_down >= h-1 or x_left < 0 or x_right >= w-1:
                continue

            else:
                intensity = ((1 - s) * (1 - t) * src[y_up, x_left]) \
                            + (s * (1 - t) * src[y_up, x_right]) \
                            + ((1 - s) * t * src[y_down, x_left]) \
                            + (s * t * src[y_down, x_right])

                dst[row, col] = intensity
    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # src의 크기는 512 x 512

    scale = 1/2
    #이미지 크기  축소
    my_dst_mini = my_bilinear(src, scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 확대 즉 원상 복구
    my_dst = my_bilinear(my_dst_mini, 1/scale)
    my_dst = my_dst.astype(np.uint8)


    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear 1/7-> restoration', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()



