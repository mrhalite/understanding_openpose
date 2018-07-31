import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from scipy.ndimage.filters import maximum_filter
import math
from collections import defaultdict
import itertools
from enum import Enum
import time

NMS_Threshold = 0.1
InterMinAbove_Threshold = 6
Inter_Threashold = 0.1
Min_Subset_Cnt = 4
Min_Subset_Score = 0.8
Max_Human = 96

# heatmap의 조합과 paf의 조합은 다음 그림을 참조
# https://cdn-images-1.medium.com/max/1600/1*BJQRrQGuW8VLH8MfA9ZngQ.png

class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

# heatmap에서 서로 이어져야 하는 것들의 index 조합
CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19

# paf에서 서로 이어져야 하는 것들의 index 조합
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
    (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
 ]  # = 19

CocoPairsRender = CocoPairs[:-2]
CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def nsm(hm):
    # non-maximum suppression으로 peak만 남긴다. 사람이 여럿인 경우 peak가 여러 개가 된다.
    hm = hm * (hm == maximum_filter(hm, footprint=np.ones((5, 5))))
    return hm

# score는 line integral 값으로 얻는데, 두 점 사이의 paf와 vector를 곱해 얻는다.
# paf가 두 점 사이의 흐름?의 정도를 나타내는데 이 흐름의 정도가 크다는 소리는 연결되어 있을 가능성이 높다는 소리다.
# https://cdn-images-1.medium.com/max/1600/1*Ce7iDD5ac7i26YXqoLQPyw.png 의 수식을 참조할 것
def get_score(x1, y1, x2, y2, pafMaxX, pafMaxY):
    num_inter = 10
    dx, dy = x2 - x1, y2 - y1
    normVec = math.sqrt(dx ** 2 + dy ** 2) # 두점 사이의 vector
    
    if normVec < 1e-4: # 같은 점이거나 거리가 가까우면 score는 0이다
        return 0.0, 0
    
    vx, vy = dx / normVec, dy / normVec

    # x1과 x2가 다르면 x1에서 <x2까지 dx/num_inter 간격으로 num_inter만큼의 숫자를 만든다. 같으면 x1으로 num_inter만큼 채운다.
    xs = np.arange(x1, x2, dx / num_inter) if x1 != x2 else np.full((num_inter, ), x1)
    ys = np.arange(y1, y2, dy / num_inter) if y1 != y2 else np.full((num_inter, ), y1)
    # xs의 값 들을 반올림 하고 np.int8 타입으로 변환한다.
    # x1, x2 사이의 간격이 pair의 index 차이이고 heatmap은 19까지, paf는 38까지 이므로 int8 타입으로 다 커버 되므로 int8 타입으로 만든 것이다.
    xs = (xs + 0.5).astype(np.int8)
    ys = (ys + 0.5).astype(np.int8)

    # without vectorization
    pafXs = np.zeros(num_inter)
    pafYs = np.zeros(num_inter)
    for idx, (mx, my) in  enumerate(zip(xs, ys)):
        # pafMaxX, pafMaxY는 전체 paf의 shape인 38xInt(H/8)xInt(W/8) 중에 첫 index 38의 값에 대해 xindx, yidx를 적용해
        # Int(H/8)xInt(W/8) 크기의 paf 값이 들어있는 array가 넘어 온다.
        pafXs[idx] = pafMaxX[my][mx]
        pafYs[idx] = pafMaxY[my][mx]

    # 수식에 따라 pafX, pafY와 vx, vy를 곱한 값을 구한다.
    local_scores = pafXs * vx + pafYs * vy
    # 곱한 값이 0.1이상인 경우만 찾고 그 index의 값은 0(False)또는 1(True)가 된다.
    thidxs = local_scores > Inter_Threashold

    # return의 첫번째 값은 local_score가 0.1이상인 값 들만 모두 합한 값이고
    # 두번째 값은 local_score가 0.1이상인 개수를 나타낸다.
    return sum(local_scores * thidxs), sum(thidxs)

def estimate_pose_pair(coords, partIdx1, partIdx2, pafX, pafY):
    connection_temp = []
    # heatmap에서 pair인 두 index에 해당하는 nms 결과 peak 들을 추출
    peak_coord1, peak_coord2 = coords[partIdx1], coords[partIdx2]

    # s = time.perf_counter()

    # heatmap index가 2개 전달되는데 각 heatmap의 
    # heatmap pair 중 첫번째 것에 대해 peak 개수 만큼 for loop를 돈다
    for idx1, (y1, x1) in enumerate(zip(peak_coord1[0], peak_coord1[1])):
        # heatmap pair 중 두번째 것에 대해 peak 개수 만큼 for loop를 돈다
        for idx2, (y2, x2) in enumerate(zip(peak_coord2[0], peak_coord2[1])):
            # 두 점 사이의 line integral 값을 얻는다.
            # 두 점을 직선으로 연결하고 그 사이의 paf 값을 더한다. (여기에서는 10개로 나누어 계산)
            # 연결된 점이라면 score나 count의 값이 크게 된다.
            score, count = get_score(x1, y1, x2, y2, pafX, pafY)

            if (partIdx1, partIdx2) in [(2, 3), (3, 4), (5, 6), (6, 7)]: # arms
                # 팔의 경우 count가 3개 이하거나 score가 0이하이면 무시
                if count < InterMinAbove_Threshold // 2 or score <= 0.0:
                    continue
            elif count < InterMinAbove_Threshold or score <= 0.0:
                # 팔 이외의 경우는 count가 6이하거나 score가 0이하면 무시
                continue

            # 두 점 사이의 연결이 인정되는 것들은 추가
            connection_temp.append({
                'score': score,
                'coord_p1': (x1, y1),
                'coord_p2': (x2, y2),
                'idx': (idx1, idx2),
                'partIdx': (partIdx1, partIdx2),
                'uPartIdx': ('{}-{}-{}'.format(x1, y1, partIdx1), '{}-{}-{}'.format(x2, y2, partIdx2))
            })
    
    # e = time.perf_counter()
    # print(e-s)

    # s = time.perf_counter()

    # 구해진 connection 들 중에서 실제 connection 만을 남긴다
    # 남기는 기준은 score가 높은 것부터 남기고 낮은 것은 없앤다.
    connection = []
    used_idx1, used_idx2 = [], []
    # 위에서 구한 connection을 score가 높은 것부터 sorting해 실제 connection을 구한다.
    for conn_candidate in sorted(connection_temp, key=lambda x: x['score'], reverse=True):
        # 이미 사용된 connection이면 무시
        if conn_candidate['idx'][0] in used_idx1 or conn_candidate['idx'][1] in used_idx2:
            continue
        # 사용되지 않은 connection은 등록
        connection.append(conn_candidate)
        used_idx1.append(conn_candidate['idx'][0])
        used_idx2.append(conn_candidate['idx'][1])

    # e = time.perf_counter()
    # print(e-s)

    return connection

def human_conns_to_human_parts(human_conns, heatMat):
    human_parts = defaultdict(lambda: None)
    for conn in human_conns:
        human_parts[conn['partIdx'][0]] = (
            conn['partIdx'][0], # part index
            (conn['coord_p1'][0] / heatMat.shape[2], conn['coord_p1'][1] / heatMat.shape[1]), # relative coordinates
            heatMat[conn['partIdx'][0], conn['coord_p1'][1], conn['coord_p1'][0]] # score
            )
        human_parts[conn['partIdx'][1]] = (
            conn['partIdx'][1],
            (conn['coord_p2'][0] / heatMat.shape[2], conn['coord_p2'][1] / heatMat.shape[1]),
            heatMat[conn['partIdx'][1], conn['coord_p2'][1], conn['coord_p2'][0]]
            )
    return human_parts

def estimate_pose(heatMat, pafMat):
    # paf.shape=(38, int(H/8), int(W/8))
    # heatmap.shape=(19, int(H/8), int(W/8))
    paf = np.rollaxis(np.squeeze(pafMat), 2, 0)
    heatmap = np.rollaxis(np.squeeze(heatMat), 2, 0)

    # # reliability issue.
    # heatmap = heatmap - heatmap.min(axis=1).min(axis=1).reshape(19, 1, 1)
    # heatmap = heatmap - heatMat.min(axis=2).reshape(19, heatmap.shape[1], 1)

    # get peak points from heatmap by NSM
    coords = []
    for hm in heatmap[:-1]:
        # non-maximum suppression
        hm = nsm(hm)
        # nsm 결과에서 찾은 peak 들의 row, col 좌표를 얻는다
        # [[x1, x2, ...], [y1, y2, ...]]과 같은 결과가 append됨.
        # coords는 (18, 2, ?)의 shape을 갖는다.
        # coords[:, :, 0]는 row 값으로 y를 의미하고, coords[:, :, 1]는 col 값으로 x를 의미함
        coords.append(np.where(hm > 0))

    # 정의해 놓은 connection pair 만큼 반복하면서 모든 connection을 구해 connection_all에 추가 한다.
    connection_all = []
    for (idx1, idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
        # 현재 pair에 해당하는 모든 connection을 찾는다.
        connection = estimate_pose_pair(coords, idx1, idx2, paf[paf_x_idx], paf[paf_y_idx])
        # connection_all에 추가 (append가 아님)
        connection_all.extend(connection)

    # 한 사람, 한 사람에 속하는 connection을 모아 준다.
    conns_by_human = dict()
    for idx, c in enumerate(connection_all):
        conns_by_human['human_%d' % idx] = [c] # 처음에 모든 connection은 다른 사람으로 취급됨

    # defaultdict()는 키값에 기본값을 정의하고 키값이 없어도 에러를 출력하지 않고 기본 값을 출력한다.
    no_merge_cache = defaultdict(list)
    empty_set = set()
    while True:
        is_merged = False
        # .combinations()는 주어진 iterable에서 주어진 개수 만큼의 combination을 만들어 준다. ('ABCD', 2)는 AB AC AD BC BD CD를 만들어 낸다.
        # .keys()는 dictionary에서 key 값들만 얻어 낸다. 여기서는 'human_0'와 같은 값을 만들어 준다.
        for h1, h2 in itertools.combinations(conns_by_human.keys(), 2): # #1의 출력 결과물을 보면 이해가 빠름
            # h1과 h2가 같으면 무시
            if h1 == h2:
                continue
            # h2가 no_merge_cache에 있으면 무시
            if h2 in no_merge_cache[h1]:
                continue
            # c1, c2는 estimate_pose_pair에서 추가된 dict로
            # uPartIdx가 connection pair 정보를 담고 있는데
            # 예를 들어 왼쪽 어깨 연결인 (1, 2)는 오른쪽 어깨 연결인 (1, 5)와 1이 서로 같다.
            # set() & set()에서 이 연결 pair중 같은 값이 있는지를 검사해
            # 같은 값이 있으면 연결되는 것으로 보고 연결점 밑으로 옮겨 주는 것이다.
            for c1, c2 in itertools.product(conns_by_human[h1], conns_by_human[h2]): # #1의 출력 결과물을 보면 이해가 빠름
                if set(c1['uPartIdx']) & set(c2['uPartIdx']) != empty_set:
                    # 연결 점이 있다
                    is_merged = True
                    # h1 밑으로 h2를 넣어 준다
                    conns_by_human[h1].extend(conns_by_human[h2])
                    # h2는 원래 것에서 제거
                    conns_by_human.pop(h2)
                    break
            if is_merged:
                # 연결된 것이 있으면 연결되지 않은 것을 나타내는 no_merge_cache에서 제거
                no_merge_cache.pop(h1, None)
                break
            else:
                # 연결된 것이 없으면 no_merge_cache에 추가
                no_merge_cache[h1].append(h2)

        if not is_merged:
            break

    # 연결점이 4개 이상(적어도 2개의 연결이 있음을 의미), score가 0.8 이상이 포함된 것에 대해서만 남긴다
    conns_by_human = {h: conns for (h, conns) in conns_by_human.items() if len(conns) >= Min_Subset_Cnt}
    conns_by_human = {h: conns for (h, conns) in conns_by_human.items() if max([conn['score'] for conn in conns]) >= Min_Subset_Score}

    humans = [human_conns_to_human_parts(human_conns, heatmap) for human_conns in conns_by_human.values()]
    return humans

def read_imgfile(path, width, height):
    img = cv2.imread(path)
    val_img = preprocess(img, width, height)
    # val_img = img
    return val_img

def preprocess(img, width, height):
    val_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 reads in BGR format
    # val_img = img
    val_img = cv2.resize(val_img, (width, height)) # each net accept only a certain size
    # val_img = cv2.resize(img, (width, height)) # each net accept only a certain size
    # val_img = val_img.reshape([1, height, width, 3])
    # val_img = val_img.astype(float)
    # val_img = val_img * (2.0 / 255.0) - 1.0 # image range from -1 to +1
    return val_img

def draw_humans(img, human_list):
    img_copied = np.copy(img)
    image_h, image_w = img_copied.shape[:2]
    centers = {}
    for human in human_list:
        part_idxs = human.keys()

        # draw point
        for i in range(CocoPart.Background.value):
            if i not in part_idxs:
                continue

            part_coord = human[i][1]
            center = (int(part_coord[0] * image_w + 0.5), int(part_coord[1] * image_h + 0.5))
            centers[i] = center
            img_copied = cv2.circle(img_copied, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in part_idxs or pair[1] not in part_idxs:
                continue

            img_copied = cv2.line(img_copied, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return img_copied

def test_keras_openpose(fileName='test5.jpg'):
    # load input image
    # W=width, H=height
    # sourceMat = cv2.imread(fileName)
    sourceMat = read_imgfile(fileName, 656, 368)
    
    # load model
    model = load_model('keras_openpose_trained_model.hd5')

    # get paf, heatmap
    # pred[0]=paf, pred[1]=heatmap
    pred = model.predict(np.array([sourceMat]))

    humans = estimate_pose(pred[1], pred[0])

    drawMat = cv2.imread(fileName)
    drawMat = draw_humans(drawMat, humans)
    cv2.imshow('result', drawMat)
    cv2.waitKey()
    
if __name__ == '__main__':
    test_keras_openpose()


# https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-1-7dd4ca5c8027


''' #1

human_0 human_1
{'score': 9.237156950751992, 'coord_p1': (38, 29), 'coord_p2': (33, 30), 'idx': (52, 65), 'pardIdx': (1, 2), 'uPartIdx': ('38-29-1', '33-30-2')} {'score': 7.194790899841582, 'coord_p1':(110, 30), 'coord_p2': (105, 32), 'idx': (55, 73), 'pardIdx':(1, 2), 'uPartIdx': ('110-30-1', '105-32-2')}
human_0 human_2
{'score': 9.237156950751992, 'coord_p1': (38, 29), 'coord_p2': (33, 30), 'idx': (52, 65), 'pardIdx': (1, 2), 'uPartIdx': ('38-29-1', '33-30-2')} {'score': 6.41193928632018, 'coord_p1': (135, 30), 'coord_p2': (132, 29), 'idx': (57, 63), 'pardIdx': (1, 2), 'uPartIdx': ('135-30-1', '132-29-2')}
human_0 human_3
{'score': 9.237156950751992, 'coord_p1': (38, 29), 'coord_p2': (33, 30), 'idx': (52, 65), 'pardIdx': (1, 2), 'uPartIdx': ('38-29-1', '33-30-2')} {'score': 8.872322948236887, 'coord_p1':(38, 29), 'coord_p2': (43, 28), 'idx': (52, 64), 'pardIdx': (1, 5), 'uPartIdx': ('38-29-1', '43-28-5')}

첫번째 loop에서 human_0과 human_3의 연결점이 발견되어 human_0에 human_3이 연결된다.
다음에는 human_0에 정보가 2개 들어 있으므로 각각에 대해 나머지와 product를 진행해 2개씩 검사를 한다.

human_0 human_1
{'score': 9.237156950751992, 'coord_p1': (38, 29), 'coord_p2': (33, 30), 'idx': (52, 65), 'pardIdx': (1, 2), 'uPartIdx': ('38-29-1', '33-30-2')} {'score': 7.194790899841582, 'coord_p1':(110, 30), 'coord_p2': (105, 32), 'idx': (55, 73), 'pardIdx':(1, 2), 'uPartIdx': ('110-30-1', '105-32-2')}
{'score': 8.872322948236887, 'coord_p1': (38, 29), 'coord_p2': (43, 28), 'idx': (52, 64), 'pardIdx': (1, 5), 'uPartIdx': ('38-29-1', '43-28-5')} {'score': 7.194790899841582, 'coord_p1':(110, 30), 'coord_p2': (105, 32), 'idx': (55, 73), 'pardIdx':(1, 2), 'uPartIdx': ('110-30-1', '105-32-2')}
human_0 human_2
{'score': 9.237156950751992, 'coord_p1': (38, 29), 'coord_p2': (33, 30), 'idx': (52, 65), 'pardIdx': (1, 2), 'uPartIdx': ('38-29-1', '33-30-2')} {'score': 6.41193928632018, 'coord_p1': (135, 30), 'coord_p2': (132, 29), 'idx': (57, 63), 'pardIdx': (1, 2), 'uPartIdx': ('135-30-1', '132-29-2')}
{'score': 8.872322948236887, 'coord_p1': (38, 29), 'coord_p2': (43, 28), 'idx': (52, 64), 'pardIdx': (1, 5), 'uPartIdx': ('38-29-1', '43-28-5')} {'score': 6.41193928632018, 'coord_p1': (135, 30), 'coord_p2': (132, 29), 'idx': (57, 63), 'pardIdx': (1, 2), 'uPartIdx': ('135-30-1', '132-29-2')}
human_0 human_4
{'score': 9.237156950751992, 'coord_p1': (38, 29), 'coord_p2': (33, 30), 'idx': (52, 65), 'pardIdx': (1, 2), 'uPartIdx': ('38-29-1', '33-30-2')} {'score': 5.717743791503171, 'coord_p1':(110, 30), 'coord_p2': (116, 29), 'idx': (55, 68), 'pardIdx':(1, 5), 'uPartIdx': ('110-30-1', '116-29-5')}
{'score': 8.872322948236887, 'coord_p1': (38, 29), 'coord_p2': (43, 28), 'idx': (52, 64), 'pardIdx': (1, 5), 'uPartIdx': ('38-29-1', '43-28-5')} {'score': 5.717743791503171, 'coord_p1':(110, 30), 'coord_p2': (116, 29), 'idx': (55, 68), 'pardIdx':(1, 5), 'uPartIdx': ('110-30-1', '116-29-5')}
human_0 human_5
{'score': 9.237156950751992, 'coord_p1': (38, 29), 'coord_p2': (33, 30), 'idx': (52, 65), 'pardIdx': (1, 2), 'uPartIdx': ('38-29-1', '33-30-2')} {'score': 4.052605987310321, 'coord_p1':(135, 30), 'coord_p2': (145, 26), 'idx': (57, 60), 'pardIdx':(1, 5), 'uPartIdx': ('135-30-1', '145-26-5')}
{'score': 8.872322948236887, 'coord_p1': (38, 29), 'coord_p2': (43, 28), 'idx': (52, 64), 'pardIdx': (1, 5), 'uPartIdx': ('38-29-1', '43-28-5')} {'score': 4.052605987310321, 'coord_p1':(135, 30), 'coord_p2': (145, 26), 'idx': (57, 60), 'pardIdx':(1, 5), 'uPartIdx': ('135-30-1', '145-26-5')}
human_0 human_6
{'score': 9.237156950751992, 'coord_p1': (38, 29), 'coord_p2': (33, 30), 'idx': (52, 65), 'pardIdx': (1, 2), 'uPartIdx': ('38-29-1', '33-30-2')} {'score': 9.553886248112386, 'coord_p1':(33, 30), 'coord_p2': (36, 39), 'idx': (65, 63), 'pardIdx': (2, 3), 'uPartIdx': ('33-30-2', '36-39-3')}

'''