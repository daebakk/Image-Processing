import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):

    # zero padding인 경우
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        #up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        #down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]
        #left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        #right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1: p_w + w]

    else:
        print('zero padding')

    return pad_img


def my_filtering(src, ftype, fshape, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (fshape[0]//2, fshape[1]//2), pad_type)
    dst = np.zeros((h, w))

    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                        #
        # 꼭 한줄로 완성할 필요 없음                           #
        ###################################################
        mask = np.ones(fshape) / (fshape[0] * fshape[1])
        #mask 확인
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                       #
        # 꼭 한줄로 완성할 필요 없음                          #
        ##################################################

        # size가 fshape이고 정중앙 값이 2인 filter
        filter = np.zeros(fshape)
        filter[fshape[0] // 2,fshape[1] // 2] = 2
        # size가 fshape인 average filter
        average_filter = np.ones(fshape) / (fshape[0] * fshape[1])
        # sharpening filter는 위에서 두 필터의 차이다.
        mask = filter - average_filter
        #mask 확인
        print(mask)


    #########################################################
    # TODO                                                  #
    # dst 완성                                               #
    # dst : filtering 결과 image                             #
    # 꼭 한줄로 완성할 필요 없음                                 #
    #########################################################

    (f_h,f_w) = fshape
    for row in range(h):
        for col in range(w):
            dst[row,col] = np.sum(src_pad[row:f_h + row,col:f_w + col] * mask)
            # overflow 문제 해결
            if dst[row,col] >= 255:
                dst[row,col] = 255
            if dst[row,col] <= 0:
                dst[row,col] = 0

    dst = (dst+0.5).astype(np.uint8)
    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # repetition padding test
    #rep_test = my_padding(src, (20,20),'repetition')

    # 3x3 filter
    dst_average = my_filtering(src, 'average', (3,3))
    dst_sharpening = my_filtering(src, 'sharpening', (3,3))

    #원하는 크기로 설정
    #dst_average = my_filtering(src, 'average', (5,7))
    #dst_sharpening = my_filtering(src, 'sharpening', (7,3))

    # 11x13 filter
    #dst_average = my_filtering(src, 'average', (11,13), 'repetition')
    #dst_sharpening = my_filtering(src, 'sharpening', (11,13),'repetition')

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    #cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
