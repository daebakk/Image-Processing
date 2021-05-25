import cv2
import numpy as np


def my_padding(src, pad_shape, pad_type='zero'):

    # zero padding인 경우
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src
    
    if pad_type == 'repetition':
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        #up
        pad_img[:p_h, p_w:p_w + w] = 1
        #down
        pad_img[p_h + h:, p_w:p_w + w] = 1
        #left
        pad_img[:, :p_w] = 1
        #right
        pad_img[:, p_w + w:] = 1

    return pad_img


def dilation(B, S):
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                           #
    ###############################################

    # 가장 자리 처리를 위해 'zero' padding을 실시
    s_h,s_w = S.shape
    s_h = s_h // 2
    s_w = s_w // 2
    pad_B = my_padding(B,(s_h,s_w),'zero')
    h,w = B.shape

    # 1. B에서 객체에 해당하는 부분 즉 값이 1인 영역만 따로 좌표로 골라내기
    object_coordinates = []
    for row in range(h):
        for col in range(w):
            if B[row,col] == 1:
                object_coordinates.append((row,col))

    # 2. 1에서 구한 좌표를 하나씩 이동해가면서 Dilation 작업 수행
    for n in range(len(object_coordinates)):
        (row,col) = object_coordinates[n]
        #print('현재 진행 dilation 좌표 ({},{})'.format(row,col))
        for i in range(-1,2):
            for j in range(-1,2):
                # dilation 하기 전 이미 값이 채워진 경우는 dilation을 하지 않는다.
                if pad_B[(row + s_h) + i,(col + s_w) + j] == S[i,j]:
                    continue
                else:
                    pad_B[(row + s_h) + i,(col + s_w) + j] = S[i,j]

    # padding에서 가장 자리 부분만 제외
    dst = pad_B[s_h:s_h+h, s_w:s_w+w]
    return dst

def erosion(B, S):
    ###############################################
    # TODO                                        #
    # erosion 함수 완성                            #
    ###############################################
    # 가장 자리 처리를 위해 zero padding을 실시
    s_h,s_w = S.shape
    s_h = s_h // 2
    s_w = s_w // 2

    pad_B = my_padding(B,(s_h,s_w),'zero')
    pad_copy = pad_B.copy()
    h,w = B.shape

    # 1. B에서 객체에 해당하는 부분 즉 값이 1인 영역만 따로 좌표로 골라내기
    object_coordinates = []
    for row in range(h):
        for col in range(w):
            if B[row,col] == 1:
                object_coordinates.append((row,col))

    # 2. 1에서 구한 좌표를 하나씩 이동해가면서 Erosion 작업 수행
    for n in range(len(object_coordinates)):
        (row,col) = object_coordinates[n]

        if np.all(pad_copy[((row + s_h)-1):((row + s_h)+2),((col + s_w)-1):((col + s_w)+2)] == S):
            pad_B[(row + s_h),(col + s_w)] = 1
        else:
            pad_B[(row + s_h), (col + s_w)] = 0
    # padding에서 가장 자리 부분만 제외
    dst = pad_B[s_h:s_h+h, s_w:s_w+w]
    return dst

def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                            #
    ###############################################

    # opening 은 erosion을 수행후 그다음 수행된 결과에 dilation 수행
    dst = erosion(B,S)
    dst = dilation(dst,S)
    return dst

def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                            #
    ###############################################
    # closing 은 dilation을 수행 후 그다음 수행된 결과에 erosion 수행

    dst = dilation(B,S)
    dst = erosion(dst,S)
    return dst


if __name__ == '__main__':

    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


    cv2.imwrite('morphology_B.png', (B*255).astype(np.uint8))

    print('before dilation : \n{}'.format(B))
    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print('after dilation : \n{}'.format(img_dilation))
    cv2.imwrite('morphology_dilation.png', img_dilation)


    print('before erosion : \n{}'.format(B))
    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print('after dilation : \n{}'.format(img_erosion))
    cv2.imwrite('morphology_erosion.png', img_erosion)



    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)


    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)


