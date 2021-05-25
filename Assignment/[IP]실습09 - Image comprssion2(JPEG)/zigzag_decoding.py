import numpy as np


def zigzag_decoding(block,block_size=8):

    dst = np.zeros((block_size,block_size))
    # 좌표 초기화
    (y,x) = (0,0)
    # 왼쪽 위 삼각형(minor diagonal 포함)
    for i in range(1,block_size+1):
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

    for i in range(block_size-1,0,-1):

        if i % 2 == 0:
            y = block_size - i
            x = block_size - 1
            n = block_size - i
            for j in range(i):
                idx = int(int((block_size * (block_size - 1)) / 2) + block_size + 8 * (n - 1) - (((n - 1) * n) / 2))
                idx += j
                dst[y+j][x-j] = block[idx]
        else:
            y = block_size - 1
            x = block_size - i
            n = block_size - i
            for j in range(i):
                idx = int(int((block_size * (block_size - 1)) / 2) + block_size + 8 * (n - 1) - (((n - 1) * n) / 2))
                idx += j
                dst[y-j][x+j] = block[idx]
    return dst



def zig_zag(block):

    (h,w) = block.shape
    lines = [[] for i in range(h + w - 1)]
    print(lines)

    for y in range(h):
        for x in range(w):
            i = y + x
            if (i % 2 == 0):
                lines[i].insert(0, block[y][x])
            else:
                lines[i].append(block[y][x])

    zig_zag_list = [coefficient for line in lines for coefficient in line]
    print('zigzag list : \n {}'.format(zig_zag_list))
    print('zigzag len : {}'.format(len(zig_zag_list)))

    """
    # EOB(End of Block) 삽입
    EOB_index = len(zig_zag_list) # 초기화
    for i in range(len(zig_zag_list),0,-1):
        if zig_zag_list[i-1] == 0:
            continue
        else:
            EOB_index = i
            break
    del zig_zag_list[i:]
    zig_zag_list.append('EOB')
    print('zigzag list(EOB) : \n {}'.format(zig_zag_list))
    """
    return zig_zag_list

if __name__ == '__main__':


    block = np.array(
        [[-26, -3, -6, 2, 2, 0, 0, 0],
         [1, -2, -4, 0, 0, 0, 0, 0],
         [-3, 1, 5, -1, -1, 0, 0, 0],
         [-4, 1, 2, -1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 45, 0, 0],
         [0, 0, 0, 45, 0, 32, 0, 50],
         [0, 24, 0, 21, 0, 33, 0, 0]]
    )
    print(block)

    list = zig_zag(block)
    #print(list)
    list = zigzag_decoding(list,8)
    print(list)
