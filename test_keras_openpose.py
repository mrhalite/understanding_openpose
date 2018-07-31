import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter

def makePairs(hm):
    pairs = np.zeros((17, 2, 2), dtype=np.int)
    
    pairs[0] = np.array([hm[0], hm[1]])

    pairs[1] = np.array([hm[1], hm[2]])
    pairs[2] = np.array([hm[2], hm[3]])
    pairs[3] = np.array([hm[3], hm[4]])

    pairs[4] = np.array([hm[1], hm[5]])
    pairs[5] = np.array([hm[5], hm[6]])
    pairs[6] = np.array([hm[6], hm[7]])

    pairs[7] = np.array([hm[1], hm[8]])
    pairs[8] = np.array([hm[8], hm[9]])
    pairs[9] = np.array([hm[9], hm[10]])

    pairs[10] = np.array([hm[1], hm[11]])
    pairs[11] = np.array([hm[11], hm[12]])
    pairs[12] = np.array([hm[12], hm[13]])

    pairs[13] = np.array([hm[0], hm[14]])
    pairs[14] = np.array([hm[14], hm[16]])

    pairs[15] = np.array([hm[0], hm[15]])
    pairs[16] = np.array([hm[15], hm[17]])

    return pairs

def nsm(hm):
    # non-maximum suppression으로 peak만 남긴다. 사람이 여럿인 경우 peak가 여러 개가 된다.
    hm = hm * (hm == maximum_filter(hm, footprint=np.ones((5, 5))))
    return hm

def test_keras_openpose():
    # load input image
    # W=width, H=height
    sourceMat = cv2.imread('test5.jpg')
    
    # load model
    model = load_model('keras_openpose_trained_model.hd5')

    # get paf, heatmap
    # pred[0]=paf, pred[1]=heatmap
    pred = model.predict(np.array([sourceMat]))

    # 첫 dimension을 제거
    # heatmap.shape=(1, int(H/8), int(W/8), 19)
    heatmap = np.squeeze(pred[1])

    # display heatmap
    grayMat = cv2.cvtColor(sourceMat, cv2.COLOR_BGR2GRAY)
    grayMat = cv2.cvtColor(grayMat, cv2.COLOR_GRAY2BGR)
    maxHeatmap = np.zeros((heatmap.shape[2] - 1, 2))

    # heatmap에 점을 그려준다. 18번째까지만 그린다.
    for i in range(heatmap.shape[2] - 1):
        hm = heatmap[:, :, i]

        # heatmap은 입력 image를 w,h 각각 1/8한 크기이다
        hm = cv2.resize(hm, (grayMat.shape[1], grayMat.shape[0]), interpolation=cv2.INTER_CUBIC)
        # find maximum position
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        maxHeatmap[i] = np.array([y, x])

        if i == 0:
            c = (0, 0, 255)
        elif i < (heatmap.shape[2] - 1):
            c = (0, 255, 0)
        else:
            c = (255, 0, 0)
        grayMat = cv2.circle(grayMat, (x, y), 2, c, thickness=2)

    # pair 간에 선을 이어준다
    pairs = makePairs(maxHeatmap)
    for i in range(pairs.shape[0]):
        p1 = (pairs[i, 0, 1], pairs[i, 0, 0])
        p2 = (pairs[i, 1, 1], pairs[i, 1, 0])
        grayMat = cv2.line(grayMat, p1, p2, (0, 255, 0), thickness=1)

    cv2.imshow('heatmap', grayMat)
    cv2.waitKey()

if __name__ == '__main__':
    test_keras_openpose()


# https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-1-7dd4ca5c8027