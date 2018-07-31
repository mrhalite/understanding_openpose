from optparse import OptionParser
import os
import cv2
from keras.models import load_model
from test_keras_openpose_2 import estimate_pose, draw_humans
import numpy as np
import time
import sys

def wk(waitTime=1, char='q'):
    c = cv2.waitKey(waitTime) & 0xFF
    if c == ord(char):
        return True
    else:
        return False

def test_keras_openpose_3(inputFileName, outputFileName):
    # load model
    model = load_model('keras_openpose_trained_model.hd5')

    videoCapture = cv2.VideoCapture(os.path.abspath(os.path.expanduser(inputFileName)))
    if not videoCapture.isOpened():
        print('Input file not opened.')
        return

    frameCount = 0

    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS) + 0.5)
    total_frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Input : {0}x{1}, {2} fps, {3} frames'.format(width, height, fps, total_frames))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(os.path.abspath(os.path.expanduser(outputFileName)), fourcc, fps, (width, height))
    if not videoWriter.isOpened():
        print('Output file not opened.')
        videoCapture.release()
        return

    while True:
        success, image = videoCapture.read()

        if not success:
            break

        print('frame={}/{}'.format(frameCount, total_frames), end='')
        sys.stdout.flush()

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (656, 368))

        s = time.perf_counter()
        pred = model.predict(np.array([img]))
        e = time.perf_counter()
        print('_{0:.4f}'.format(e - s), end='')
        sys.stdout.flush()

        s = time.perf_counter()
        humans = estimate_pose(pred[1], pred[0])
        e = time.perf_counter()
        print('_{0:.4f}'.format(e - s), end='')
        sys.stdout.flush()

        s = time.perf_counter()
        drawMat = draw_humans(image, humans)
        e = time.perf_counter()
        print('_{0:.4f}'.format(e - s), end='')
        sys.stdout.flush()

        s = time.perf_counter()
        videoWriter.write(drawMat)
        e = time.perf_counter()
        print('_{0:.4f}'.format(e - s), end='')
        sys.stdout.flush()

        frameCount += 1

        print('')
        sys.stdout.flush()

        cv2.imshow('resutl', drawMat)
        if wk() == True:
            break

    videoCapture.release()
    videoWriter.release()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--input", dest="input", default=None, help="Input movie file name")
    parser.add_option("--output", dest="output", default=None, help="Output file name")
    (options, args) = parser.parse_args()

    test_keras_openpose_3(options.input, options.output)
