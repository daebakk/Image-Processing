import numpy as np
import cv2

def my_normalize(src):

    dst = src.copy()
    dst *= 255
    dst = np.clip(dst,0,255)

    return dst.astype(np.uint8)


def add_gaus_noise(src,mean=0,sigma=0.1):

    #src : 0 ~ 255, dst : 0 ~ 1

    dst = src / 255
    (h,w) = dst.shape
    noise = np.random.normal(mean,sigma,size=(h,w))
    print('noise : \n{}'.format(noise))
    dst += noise

    return my_normalize(dst)



def main():


    # 과제 진행시 seed 값 변경 하지 말것
    np.random.seed(seed=100) # 값이 고정
    # 평균이 0 표준편차가 1 filter size가 5
    rand_norm = np.random.normal(0,1,size=5)
    print(rand_norm)

    rand_norm2 = np.random.normal(0,1,size=(3,3))
    print(rand_norm2)


    src = cv2.imread('Lena.png',cv2.IMREAD_GRAYSCALE)

    # I_g(x,y) = I(x,y) + N(x,y)
    dst_noise = add_gaus_noise(src,mean=0,sigma=0.1)

    cv2.imshow('original',src)
    cv2.imshow('add gaus noise',dst_noise)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
