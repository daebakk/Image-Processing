import cv2
import numpy as np


def my_padding(src, pad_shape, pad_type='zero'):
    # zero padding인 경우
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:p_h + h, p_w:p_w + w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        # up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]
        # left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        # right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1: p_w + w]
    return pad_img


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    # mask의 크기
    (m_h, m_w) = mask.shape
    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, (m_h // 2, m_w // 2), pad_type)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:m_h + row, col:m_w + col] * mask)
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
    DoG_x = DoG_x - (DoG_x.sum() / fsize ** 2)
    DoG_y = DoG_y - (DoG_y.sum() / fsize ** 2)

    # Dog_x, Dog_y shape 및 값 확인
    print('Dog_x shape  : {}'.format(DoG_x.shape))
    print('Dog_y shape  : {}'.format(DoG_y.shape))
    print('Dog_x : \n{}'.format(DoG_x))
    print('Dog_y : \n{}'.format(DoG_y))
    print('Dog_x sum : \n{}'.format(np.sum(DoG_x)))
    print('Dog_y sum : \n{}'.format(np.sum(DoG_y)))

    return DoG_x, DoG_y

# low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨

    ###########################################
    # TODO                                    #
    # apply_lowNhigh_pass_filter 완성          #
    # Ix와 Iy 구하기                            #
    ###########################################

    # Dog 방법 사용
    # DoG_x, DoG_y mask 구하기
    DoG_x, DoG_y = get_DoG_filter(fsize,sigma)
    Ix = my_filtering(src,DoG_x,'zero')
    Iy = my_filtering(src,DoG_y,'zero')

    return Ix, Iy

# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                    #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    return magnitude


# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    ###################################################
    # TODO                                            #
    # calcAngle 완성                                   #
    # angle     : ix와 iy의 angle                      #
    # e         : 0으로 나눠지는 경우가 있는 경우 방지용     #
    # np.arctan 사용하기(np.arctan2 사용하지 말기)        #
    ###################################################
    e = 1E-6

    # angle은 60분법
    # angle의 범위 : -90 ~ 90
    angle = np.rad2deg(np.arctan(Iy / (Ix + e)))
    return angle


# non-maximum suppression 수행
def non_maximum_suppression(magnitude, angle):

    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # largest_magnitude     : non_maximum_suppression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    (h, w) = magnitude.shape
    # angle의 범위 : -90 ~ 90
    larger_magnitude = np.zeros((h, w), dtype=np.uint8)
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            # gradient의 방향
            degree = angle[row, col]
            # gradient의 degree는 edge와 수직방향이다.
            if 0 <= degree and degree < 45:
                rate = np.tan(np.deg2rad(degree))
                left_magnitude = (rate) * magnitude[row - 1, col - 1] + (1 - rate) * magnitude[row, col - 1]
                right_magnitude = (rate) * magnitude[row + 1, col + 1] + (1 - rate) * magnitude[row, col + 1]
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    larger_magnitude[row, col] = magnitude[row, col]

            elif 90 >= degree and degree >= 45:
                rate = 1 / np.tan(np.deg2rad(degree))
                up_magnitude = (rate) * magnitude[row - 1, col - 1] + (1 - rate) * magnitude[row - 1, col]
                down_magnitude = (rate) * magnitude[row + 1, col + 1] + (1 - rate) * magnitude[row + 1, col]
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    larger_magnitude[row, col] = magnitude[row, col]

            elif -45 <= degree and degree < 0:
                rate = -np.tan(np.deg2rad(degree))
                left_magnitude = (rate) * magnitude[row + 1, col - 1] + (1 - rate) * magnitude[row, col - 1]
                right_magnitude = (rate) * magnitude[row - 1, col + 1] + (1 - rate) * magnitude[row, col + 1]
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    larger_magnitude[row, col] = magnitude[row, col]

            elif -45 > degree and degree >= -90:
                rate = - 1 / np.tan(np.deg2rad(degree))
                up_magnitude = (rate) * magnitude[row - 1, col + 1] + (1 - rate) * magnitude[row - 1, col]
                down_magnitude = (rate) * magnitude[row + 1, col - 1] + (1 - rate) * magnitude[row + 1, col]
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    larger_magnitude[row, col] = magnitude[row, col]

            else:
                #Ix값이 0일 경우  nan으로 처리
                print(row, col, '잘못된 입력 :  degree :', degree)

    return larger_magnitude


def find_connected_weak_edge(src, row,col,connected_weak_edge_coordinates, high_threshold, low_threshold):
    # 중복인 경우 함수 종료
    if ((row,col) in connected_weak_edge_coordinates):
        return

    # weak edge가 아닌 경우 탐색 종료
    elif (src[row,col] > high_threshold) or ( src[row,col] < low_threshold):
        return
    else:

        (h,w) = src.shape
        # 이웃한 8개 좌표
        #-------------------
        #| 1   | 2     |3
        #| 4   | 5     |6
        #| 7   | 8     |9

        # weak edge인 경우 값 추가
        if src[row,col] <= high_threshold and src[row,col] >= low_threshold:
            connected_weak_edge_coordinates.append((row,col))
            # 1번 좌표
            if row > 0 and col > 0:
                find_connected_weak_edge(src, row - 1, col - 1, connected_weak_edge_coordinates, high_threshold,low_threshold)
            # 2번 좌표
            if row > 0:
                find_connected_weak_edge(src, row - 1, col, connected_weak_edge_coordinates, high_threshold,low_threshold)
            # 3번 좌표
            if row > 0 and col < w - 1:
                find_connected_weak_edge(src, row - 1, col + 1, connected_weak_edge_coordinates, high_threshold,low_threshold)
            # 4번 좌표
            if col > 0:
                find_connected_weak_edge(src, row, col - 1, connected_weak_edge_coordinates, high_threshold,low_threshold)
            # 6번 좌표
            if col < w - 1:
                find_connected_weak_edge(src, row, col + 1, connected_weak_edge_coordinates, high_threshold,low_threshold)
            # 7번 좌표
            if row < h - 1 and col > 0:
                find_connected_weak_edge(src, row + 1, col - 1, connected_weak_edge_coordinates, high_threshold,low_threshold)
            # 8번 좌표
            if row < h - 1:
                find_connected_weak_edge(src, row + 1, col, connected_weak_edge_coordinates, high_threshold,low_threshold)
            # 9번 좌표
            if row < h - 1 and col < w - 1:
                find_connected_weak_edge(src, row + 1, col + 1, connected_weak_edge_coordinates, high_threshold,low_threshold)

def hysteresis_traking(src, weak_edge_coordinates, high_threshold, low_threshold):

    # weak edge를 전부 꺼내어 edge 유무를 판단한다.
    for i in range(len(weak_edge_coordinates)):
        connected_weak_edge_coordinates = []
        (row, col) = weak_edge_coordinates[i]

        # 위의 (row,col)점과 연결된 모든 weak edge들을 찾는 함수
        find_connected_weak_edge(src, row,col,connected_weak_edge_coordinates,high_threshold,
                                 low_threshold)

        connected_weak_edge_coordinates = list(set(connected_weak_edge_coordinates))

        # (row,col)과 연결된 모든 weak edge들에 대하여 주변 픽셀 값들을 비교
        # 근처에 strong edge가 한개라도 있으면 strong edge 255 값 채움
        # (row,col)의 이웃 좌표가 index를 초과할수 있으므로 padding 처리된 이미지에서 판단
        pad_img = my_padding(src, (1, 1), 'zero')
        for i in range(len(connected_weak_edge_coordinates)):
            (y, x) = connected_weak_edge_coordinates[i]

            neighbor = pad_img[(y + 1) - 1:(y + 1) + 2, (x + 1) - 1:(x + 1) + 2]
            if np.any(neighbor > high_threshold):
                # 연결된 모든 Edge에 대하여 strong edge로 간주
                for i in range(len(connected_weak_edge_coordinates)):
                    (y, x) = connected_weak_edge_coordinates[i]
                    src[y, x] = 255
                break

        # 위의 for문을 지난 후 연결된 모든 Edge에 대해서 strong edge가 없으면 0값 (non-edge) 채움
        if src[row, col] != 255:
            for i in range(len(connected_weak_edge_coordinates)):
                (y,x) = connected_weak_edge_coordinates[i]
                src[y,x] = 0

    return src


# double_thresholding 수행
def double_thresholding(src,high_threshold_value,low_threshold_value):
    dst = src.copy()


    dst = (((dst - np.min(dst)) / np.max(dst)) * 255).astype(np.uint8)

    (h, w) = dst.shape

    print('high threshold : {} , low threshold : {}'.format(high_threshold_value, low_threshold_value))

    ######################################################
    # TODO                                               #
    # double_thresholding 완성                            #
    # dst     : double threshold 실행 결과 이미지           #
    ######################################################

    # strong edge -> 흰색(255)
    # edge가 아니면 -> 검은색(0)
    # weak edge이면 -> 회색(128), weak edge의 좌표들만 따로 저장
    weak_edge_coordinates = []
    for row in range(h):
        for col in range(w):
            # print('current original cooridnates : ({},{})'.format(row,col))
            # strong edge -> 흰색 (255)
            if dst[row, col] > high_threshold_value:
                dst[row, col] = 255
            # edge가 아니면 -> 검은색(0)
            elif dst[row, col] < low_threshold_value:
                dst[row, col] = 0
            # weak edge면 weak edge 목록에 따로 추가
            else:
                weak_edge_coordinates.append((row, col))
    return dst, weak_edge_coordinates


def my_canny_edge_detection(src, fsize=3, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    # DoG 를 사용하여 1번 filtering
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma)

    # Ix와 Iy 시각화를 위해 임시로 Ix_t와 Iy_t 만들기
    Ix_t = np.abs(Ix)
    Iy_t = np.abs(Iy)
    Ix_t = Ix_t / Ix_t.max()
    Iy_t = Iy_t / Iy_t.max()

    cv2.imshow("Ix", Ix_t)
    cv2.imshow("Iy", Iy_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    magnitude_t = magnitude
    magnitude_t = magnitude_t / magnitude_t.max()
    cv2.imshow("magnitude", magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # non-maximum suppression 수행
    largest_magnitude = non_maximum_suppression(magnitude, angle)

    high_threshold_value, _ = cv2.threshold(largest_magnitude, 0, 255, cv2.THRESH_OTSU)
    low_threshold_value = high_threshold_value * 0.4

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    largest_magnitude_t = largest_magnitude
    largest_magnitude_t = largest_magnitude_t / largest_magnitude_t.max()
    cv2.imshow("largest_magnitude", largest_magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # double thresholding 수행
    dst, weak_edge_coordinates = double_thresholding(largest_magnitude,
                                                     high_threshold_value,low_threshold_value)
    #cv2.imshow('strong + weak edge', dst)
    dst = hysteresis_traking(dst, weak_edge_coordinates, high_threshold_value, low_threshold_value)
    return dst


def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    print('original size : {}'.format(src.shape))
    dst = my_canny_edge_detection(src)
    print('dst size : {}'.format(dst.shape))
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()