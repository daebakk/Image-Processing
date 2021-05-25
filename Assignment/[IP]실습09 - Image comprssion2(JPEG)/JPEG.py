import numpy as np
import cv2
import time



def zero_padding_for_DCT(src,n=8):

    (h,w) = src.shape

    # padding shape의 초기화
    (p_h,p_w) = (h,w)
    # n의 배수로 h,w를 늘려준다.
    if h % n != 0:
        p_h = h + n - (h % n)
    if w % n != 0:
        p_w = w + n - (w % n)

    # n의 배수로 늘려준 h,w를 size로 한다.
    dst = np.zeros((p_h,p_w))
    # 늘려준 크기에서 원본 이미지 크기에 해당하는 원본 값을 그대로 넣어준다.
    dst[:h,:w] = src
    return dst

def C1(w,n=8):

    if w == 0:
        return (1 / n) ** (0.5)
    else:
        return (2 / n) ** (0.5)

    return dst

def C2(arr,n=8):

    (h,w) = arr.shape
    arr = arr.astype(np.float32)
    for row in range(h):
        for col in range(w):
            if arr[row][col] == 0:
                arr[row][col] = (1 / n) ** (0.5)
            else:
                arr[row][col] = (2 / n) ** (0.5)
    return arr

def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance

def img2block(src, n=8):
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    ######################################

    blocks = []
    (h,w) = src.shape
    img_pad = src.copy()

    # 원본 이미지의 size가 8로 나누어 떨어지는 경우와 그렇지 않은 경우로 구분
    # 나누어 떨어지지 않으면 zero-padding을 줘서 size을 8의 배수로 맞춘다.
    # high 또는 width가  8의 배수가 아닌 경우

    if (h % n != 0) or (w % n != 0):
        img_pad = zero_padding_for_DCT(src,n)

    # 초기화
    (p_h,p_w) = img_pad.shape
    # 각 행과 열 별로 block의 개수
    # int를 쓴 이유는 소수점을 제거하기 위한 것이 아니라 data type을 int로 하기 위한 것.
    (b_h,b_w) = (int(p_h / n), int(p_w / n))
    print("b_h : {} b_w : {}".format(b_h,b_w))

    for row in range(b_h):
        for col in range(b_w):
            # block들을 1차원 list에 추가.
            blocks.append(img_pad[row*n:row*n + n,col*n:col*n + n])

    # list인 block을 numpy array로 변환
    # blocks shape : 4096 x 8 x 8
    return np.array(blocks,dtype=np.float32)


def DCT(block, n=8):
    ######################################
    # TODO                               #
    # DCT 완성                            #
    ######################################
    dst = np.zeros(block.shape)
    v,u = dst.shape

    y,x = np.mgrid[0:u,0:v]

    for v_ in range(v):
        for u_ in range(u):
            tmp = block * np.cos((2*x+1) * u_* np.pi / (2*n)) * np.cos((2*y+1) * v_* np.pi / (2*n))

            dst[v_,u_] = C1(u_,n=n) * C1(v_,n=n) * np.sum(tmp)

    return np.round(dst)

def my_zigzag_scanning(block,mode='',block_size=8):
    ######################################
    # TODO                               #
    # my_zigzag_scanning 완성             #
    ######################################
    # zigzag decoding

    if mode == 'decoding':
        # EOB를 모두 0으로 치환
        EOB_index = len(block)-1
        del block[EOB_index]
        for i in range(block_size**2 - EOB_index):
            block.append(0)

        # block은 현재 block_size * block_size개의 원소를 가지고 있는 list
        # 이 block을 (block_size ,block_size)의 원래의 2차원 배열로 만든다.
        dst = np.zeros((block_size, block_size))
        # 좌표 초기화
        (y, x) = (0, 0)
        # 왼쪽 위 삼각형(minor diagonal 포함)
        for i in range(1, block_size + 1):
            if i % 2 == 0:
                y = 0
                x = i - 1
                for j in range(i):
                    idx = int((i * (i - 1)) / 2)
                    idx += j
                    dst[y + j][x - j] = block[idx]
            else:
                y = i - 1
                x = 0
                for j in range(i):
                    idx = int((i * (i - 1)) / 2)
                    idx += j
                    dst[y - j][x + j] = block[idx]

        # 오른쪽 아래 삼각형(minor diagonal 미포함)

        for i in range(block_size - 1, 0, -1):

            if i % 2 == 0:
                y = block_size - i
                x = block_size - 1
                n = block_size - i
                for j in range(i):
                    idx = int(int((block_size * (block_size - 1)) / 2) + block_size + 8 * (n - 1) - (((n - 1) * n) / 2))
                    idx += j
                    dst[y + j][x - j] = block[idx]
            else:
                y = block_size - 1
                x = block_size - i
                n = block_size - i
                for j in range(i):
                    idx = int(int((block_size * (block_size - 1)) / 2) + block_size + 8 * (n - 1) - (((n - 1) * n) / 2))
                    idx += j
                    dst[y - j][x + j] = block[idx]
        return dst

    # zigzag encoding
    else:
        (h, w) = block.shape
        lines = [[] for i in range(h + w - 1)]


        for y in range(h):
            for x in range(w):
                i = y + x
                if (i % 2 == 0):
                    lines[i].insert(0, block[y][x])
                else:
                    lines[i].append(block[y][x])

        zig_zag_list = [elements for line in lines for elements in line]

        # EOB(End of Block) 삽입
        EOB_index = len(zig_zag_list)  # 초기화
        for i in range(len(zig_zag_list), 0, -1):
            if zig_zag_list[i - 1] == 0:
                continue
            else:
                EOB_index = i
                break
        del zig_zag_list[EOB_index:]
        zig_zag_list.append('EOB')

        return zig_zag_list


def DCT_inv(block, n = 8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################
    dst = np.zeros(block.shape)
    y,x = dst.shape
    v,u = np.mgrid[0:y,0:x]
    C_u = C2(u, n=n)
    C_v = C2(v, n=n)
    for y_ in range(y):
        for x_ in range(x):
            tmp = block * np.cos((2*x_+1) * u* np.pi / (2*n)) * np.cos((2*y_+1) * v* np.pi / (2*n)) * C_u * C_v
            dst[y_,x_] = np.sum(tmp)

    return np.round(dst)

def block2img(blocks, src_shape, n = 8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################
    (h, w) = src_shape
    # 원본 이미지의 size가 8로 나누어 떨어지는 경우와 그렇지 않은 경우로 구분
    # 나누어 떨어지지 않으면 zero-padding을 줘서 size을 8의 배수로 맞춘다.
    # high 또는 width가 8의 배수가 아닌 경우
    (p_h,p_w) = (h,w)
    if h % n != 0:
        p_h = h + n - (h % n)
    if w % n != 0:
        p_w = w + n - (w % n)

    dst = np.zeros((p_h,p_w))
    # 각 행과 열 별로 block의 개수
    (b_h, b_w) = (int(p_h / n), int(p_w / n))

    for row in range(b_h):
        for col in range(b_w):
            dst[n*row:n*row + n,n*col:n*col + n] = blocks[0]
            blocks = np.delete(blocks,0,0)

    # padding된 이미지라면 원본 사이즈 크기로 다시 되돌려 놓는다.
    dst = dst[:h,:w]
    return dst

def Encoding(src, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    # 예를들어, Lena image의 경우 512 x 512
    # 8로 나누면 64개의 block이 생김
    blocks = img2block(src, n=n)

    #subtract 128
    blocks -= 128
    print('encoding subtract dct[0] : \n{}'.format(blocks[0]))

    #DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)
    print('encoding dct[0] : \n{}'.format(blocks_dct[0]))

    #Quantization + thresholding
    Q = Quantization_Luminance()
    QnT = np.round(blocks_dct / Q)
    print('encoding Quantization[0] : \n{}'.format(QnT[0]))

    # zigzag scanning
    zz = []
    for i in range(len(QnT)):
        zz.append(my_zigzag_scanning(QnT[i]))

    return zz, src.shape

def Decoding(zigzag, src_shape, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n))
    blocks = np.array(blocks)
    print('blocks shape : {}'.format(blocks.shape))
    print('decoding zigzag blocks[0] : \n{}'.format(blocks[0]))

    # Denormalizing
    Q = Quantization_Luminance()
    blocks = blocks * Q
    print('decoding Quantization[0] : \n{}'.format(blocks[0]))

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)
    print('decoding idct[0] : \n{}'.format(blocks_idct[0]))

    # add 128
    blocks_idct += 128

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst



def main():
    start = time.time()
    src = cv2.imread('./Lena.png', cv2.IMREAD_GRAYSCALE)
    print('src min : {} max : {}'.format(np.min(src),np.max(src)))
    comp, src_shape = Encoding(src, n=8)

    # 과제의 comp.npy, src_shape.npy를 복구할 때 아래 코드 사용하기(위의 2줄은 주석처리하고, 아래 2줄은 주석 풀기)
    #comp = np.load('comp.npy', allow_pickle=True)
    #src_shape = np.load('src_shape.npy')

    recover_img = Decoding(comp, src_shape, n=8)
    total_time = time.time() - start

    print('dst shape : {}'.format(recover_img.shape))
    print('time : ', total_time)
    recover_img = np.clip(recover_img,0,255)
    recover_img = recover_img.astype(np.uint8)
    print('dst min : {} max : {}'.format(np.min(recover_img), np.max(recover_img)))
    print('normalization dst'.format(recover_img))


    if total_time > 45:
        print('감점 예정입니다.')
    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
