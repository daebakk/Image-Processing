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

    sobel_x, sobel_y = get_sobel()

    src = cv2.imread('./sobel_test.png',cv2.IMREAD_GRAYSCALE)

    # 현재 자료형 값은 float 즉 1이상은 모두 1(흰색)이고 0이하는 모두 0(검은색)이다.
    # 255로 나누어 줌으로서 원래 1~254(흰색)인 값을 검은색 또는 회색으로 바꿔준다.
    # edge 부분은 양이든 음이든 값이 크다. 따라서 대부분 흰색 아니면 검은색이다
    # edge가 아닌부분은 값이 작다. 따라서 검은색 처리를 한다.

    dst_x = my_filtering(src,sobel_x,'zero')
    dst_y = my_filtering(src,sobel_y,'zero')

    dst_x_norm = (dst_x - np.min(dst_x)) / np.max(dst_x - np.min(dst_x))
    dst_y_norm = (dst_y - np.min(dst_y)) / np.max(dst_y - np.min(dst_y))

    cv2.imshow('dst_x',dst_x / 255)
    cv2.imshow('dst_y',dst_y / 255)
    cv2.imshow('dst_x_norm', dst_x_norm)
    cv2.imshow('dst_y_norm', dst_y_norm)

    # 현재 자료형 값은 float
    abs_dst_x = np.abs(dst_x)
    abs_dst_y = np.abs(dst_y)

    cv2.imshow('abs(dst_x)',abs_dst_x / 255)
    cv2.imshow('abs(dst_y)',abs_dst_y/ 255)



    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

