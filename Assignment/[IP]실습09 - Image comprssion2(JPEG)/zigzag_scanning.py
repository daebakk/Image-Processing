import numpy as np

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
    return zig_zag_list

if __name__ == '__main__':


    block = np.array(
        [[-26, -3, -6, 2, 2, 0, 0, 0],
         [1, -2, -4, 0, 0, 0, 0, 0],
         [-3, 1, 5, -1, -1, 0, 0, 0],
         [-4, 1, 2, -1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]
    )

    list = zig_zag(block)
