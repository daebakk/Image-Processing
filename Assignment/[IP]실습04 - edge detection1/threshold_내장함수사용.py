import numpy as np
import  cv2

def main():

    src = cv2.imread('./threshold_test.png',cv2.IMREAD_GRAYSCALE)
    #ret, dst = cv2.threshold(src,150,255,cv2.THRESH_BINARY)

    # 자동으로 threshold value을 설정해주는 방법 - 내장함수 사용
    ret, dst = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    print('ret : ',ret)
    cv2.imshow('original',src)
    cv2.imshow('threshold_Test',dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()