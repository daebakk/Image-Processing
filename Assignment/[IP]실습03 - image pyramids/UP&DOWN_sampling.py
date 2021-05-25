import cv2
import numpy as np

"""
Gaussian image pyramid downsampling

1. 이미지에 gaussian filter 적용
2. blur된 이미지에서 가로와 세로를 1/2 씩 줄임
3. 짝수 번째 행과 열을 지움
"""

"""
Gaussian image pyramid upsamling

1. 이미지에서 가로와 세로를 2배 늘림
2. 픽셀값을 반복해서 빈 곳을 채움
"""


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
            sum = 0
            for m_row in range(m_h):
                for m_col in range(m_w):
                    sum += pad_img[row + m_row, col + m_col] * mask[m_row, m_col]
            dst[row, col] = sum

    return dst

def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    y, x = np.mgrid[-(msize // 2) : (msize // 2) + 1, -(msize // 2) : (msize // 2) + 1]
    '''
    y, x = np.mgrid[-1:2, -1:2]
    y = [[-1,-1,-1],
         [ 0, 0, 0],
         [ 1, 1, 1]]
    x = [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    '''

    # 2차 gaussian mask 생성
    gaus2D = 1 / ( 2 * np.pi * sigma**2) * np.exp(-((x**2 + y**2) / (2 * sigma ** 2)))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)
    return gaus2D



"""
Gaussian image pyramid downsampling

1. 이미지에 gaussian filter 적용
2. blur된 이미지에서 가로와 세로를 1/2 씩 줄임
3. 짝수 번째 행과 열을 지움
"""

def my_gaussain_downsampling(src, gap, mask, pad_type):

    (h,w) = src.shape

    # gaussian filter 적용 o
    blur_img = my_filtering(src,mask,pad_type)

    # gaussian filter 적용 x
    # blusr_img = src.copy()

    dst = np.zeros((h // gap, w // gap))
    (h_dst, w_dst) = dst.shape
    for row in range(h_dst):
        for col in range(w_dst):
            dst[row,col] = blur_img[row*gap, col*gap]

    return dst


"""
Gaussian image pyramid upsamling

1. 이미지에서 가로와 세로를 2배 늘림
2. 픽셀값을 반복해서 빈 곳을 채움
"""

def my_gaussian_upsampling(src,gap):

    (h,w) = src.shape

    dst = np.zeros((h*gap, w*gap))
    (h_dst, w_dst) = dst.shape

    for row in range(h_dst):
        for col in range(w_dst):
            dst[row,col] = src[row // gap, col // gap]

    return dst

def my_gaussian_pyramids(src, repeat, gap=2, msize=3, sigma=1, pad_type = 'zero'):

    dsts_down = []
    dsts_up = []

    # 다운 샘플링의 첫번째 인자에 src 이미지를 넣는다.
    # 즉 피라미드의 가장 low level(level0)을 의미.
    dsts_down.append(src.copy())

    gaus2D = my_get_Gaussian2D_mask(msize,sigma)

    for i in range(repeat):
        dsts_down.append(my_gaussain_downsampling(dsts_down[i],gap,gaus2D,pad_type))
        print('repeat {}'.format(i))
        print(dsts_down[i])


    # 업 샘플링의 첫번째 인자에 가장 작게 다운 샘플링된 이미지
    # 즉 high level을 의미.

    dsts_up.append(dsts_down[repeat])
    for i in range(repeat):
        dsts_up.append(my_gaussian_upsampling(dsts_up[i],gap))

    return dsts_down, dsts_up


if __name__ == '__main__':

    src = cv2.imread('Lena.png',cv2.IMREAD_GRAYSCALE)
    src = src.astype(np.float32) # 오버플로우 문제 제거

    dst_gaus_down, dst_gaus_up = my_gaussian_pyramids(src, 2, msize=3, sigma=1, pad_type='repetition')

    for i in range(len(dst_gaus_down)):
        img = dst_gaus_down[i]
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        cv2.imshow('gaussian dst %d downsampling'%i,img)
        cv2.waitKey()



    for i in range(len(dst_gaus_up)):
        img = dst_gaus_up[i]
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        cv2.imshow('gaussian dst %d upsampling'%i,img)
        cv2.waitKey()

    cv2.destroyAllWindows()







