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

def get_DoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################

    y, x = np.mgrid[-(fsize // 2): (fsize // 2) + 1, -(fsize // 2): (fsize // 2) + 1]

    DoG_x = -x / (sigma**2) * np.exp(-((x**2 + y**2) / (2 * sigma ** 2)))
    DoG_y = -y / (sigma**2) * np.exp(-((x**2 + y**2) / (2 * sigma ** 2)))


    # 필터의  합을 0으로 만들기
    DoG_x = DoG_x - (DoG_x.sum() / fsize ** 2 )
    DoG_y = DoG_y - (DoG_y.sum() / fsize ** 2)

    # Dog_x, Dog_y shape 및 값 확인
    print('Dog_x shape  : {}'.format(DoG_x.shape))
    print('Dog_y shape  : {}'.format(DoG_y.shape))
    print('Dog_x : \n{}'.format(DoG_x))
    print('Dog_y : \n{}'.format(DoG_y))

    return DoG_x, DoG_y

def main():
    src = cv2.imread('../resources/Lena.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_DoG_filter(fsize=3, sigma=1)

    ###################################################
    # TODO                                            #
    # DoG mask sigma값 조절해서 mask 만들기              #
    ###################################################
    # DoG_x, DoG_y filter 확인
    x,y = get_DoG_filter(fsize=257, sigma=30)

    # 시각화 작업
    x = ((x - np.min(x)) / np.max(x - np.min(x)) * 255).astype(np.uint8)
    y = ((y - np.min(y)) / np.max(y - np.min(y)) * 255).astype(np.uint8)

    # DoG_x, DoG_y filter 결과 : dst_x, dst_y
    dst_x = my_filtering(src, DoG_x, 'zero')
    dst_y = my_filtering(src, DoG_y, 'zero')

    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################
    dst = np.sqrt(dst_x**2 + dst_y**2)
    
    cv2.imshow('DoG_x filter', x)
    cv2.imshow('DoG_y filter', y)
    cv2.imshow('dst_x', dst_x/255)
    cv2.imshow('dst_y', dst_y/255)
    cv2.imshow('dst', dst/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

