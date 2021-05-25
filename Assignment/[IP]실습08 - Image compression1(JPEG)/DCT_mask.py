import cv2
import numpy as np

def Spatial2Frequency_mask(block, n = 8):

    # dst shape : 4 x 4
    dst = np.zeros(block.shape)
    # 4 x 4
    v, u = dst.shape
    y, x = np.mgrid[0:v, 0:u]
    # mask shape : 16 x 16
    mask = np.zeros((n*n, n*n))

    for v_ in range(v):
        for u_ in range(u):
            ##########################################################################
            # ToDo                                                                   #
            # mask 만들기                                                             #
            # mask.shape = (16x16)                                                   #
            # DCT에서 사용된 mask는 (4x4) mask가 16개 있음 (u, v) 별로 1개씩 있음 u=4, v=4  #
            # 4중 for문으로 구현 시 감점 예정                                             #
            ##########################################################################
            submask = np.cos((2*x+1) * u_* np.pi / (2*n)) * np.cos((2*y+1) * v_* np.pi / (2*n))

            # normalization
            mask[n * v_:n * v_ + n, n * u_:n * u_ + n] = my_normalize(submask)

    return mask

def my_normalize(src):
    ##############################################################################
    # ToDo                                                                       #
    # my_normalize                                                               #
    # mask를 보기 좋게 만들기 위해 어떻게 해야 할 지 생각해서 my_normalize 함수 완성해보기   #
    ##############################################################################

    # 최소값과 최대값이 같으면 minmax scaling을 하지 않고 단순히 255을 곱함
    if np.min(src) == np.max(src):
        return src * 255
    else:
        # 1. mask를 0과 1사이로 rescaling - minmax scaling
        dst = (src - np.min(src)) / (np.max(src) - np.min(src))
        # 2. 1의 결과에서 255를 곱하므로서 mask를 0과 255사이로 rescaling
        dst *= 255
    return dst

if __name__ == '__main__':
    block_size = 4
    src = np.ones((block_size, block_size))

    mask = Spatial2Frequency_mask(src, n=block_size)
    mask = mask.astype(np.uint8)
    print('final normalization mask : \n{}'.format(mask))

    #크기가 너무 작으니 크기 키우기 (16x16) -> (320x320)
    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('201804222_mask', mask)
    cv2.waitKey()
    cv2.destroyAllWindows()



