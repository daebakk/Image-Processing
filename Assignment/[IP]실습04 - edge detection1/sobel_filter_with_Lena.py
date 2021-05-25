import numpy as np
import cv2
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


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    # mask의 크기
    (m_h, m_w) = mask.shape
    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, (m_h // 2, m_w // 2), pad_type)

    print('<mask>')
    print(mask)

    # 시간을 측정할 때 만 이 코드를 사용하고 시간측정 안하고 filtering을 할 때에는
    # 4중 for문으로 할 경우 시간이 많이 걸리기 때문에 2중 for문으로 사용하기.
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row,col] = np.sum(pad_img[row:m_h + row,col:m_w + col] * mask)
    return dst


def get_sobel():
    derivative = np.array([[-1,0,1]])
    blur = np.array([[1],[2],[3]])

    x = np.dot(blur,derivative)
    y = np.dot(derivative.T,blur.T)

    return x,y


def main():

    src = cv2.imread('../resources/Lena.png',cv2.IMREAD_GRAYSCALE)
    sobel_x, sobel_y = get_sobel()

    dst_x = my_filtering(src,sobel_x,'zero')
    dst_y = my_filtering(src,sobel_y,'zero')
    dst = np.abs(dst_x) + np.abs(dst_y)
    ret, dst_threshold = cv2.threshold(dst,100,255,cv2.THRESH_BINARY)

    print('ret : ',ret)
    cv2.imshow('before threshold',dst/255)
    cv2.imshow('after threshold',dst_threshold/255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

