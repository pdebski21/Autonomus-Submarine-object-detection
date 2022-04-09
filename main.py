import numpy as np
import cv2 as cv


def captureFrame():
    cap = cv.VideoCapture(0)
    count = 0
    while cap.isOpened():
        ret,frame = cap.read()
        frame = detectCorners(frame)
        cv.imshow('window-name', frame)
        #cv.imwrite("frame%d.jpg" % count, frame)
        count = count + 1
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() # destroy all opened windows


def detectCorners(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2,3,0.04)
    dst = cv.dilate(dst,None)
    frame[dst>0.01*dst.max()]=[0, 0, 255]
    return frame


def displayFrame(frame, windowName):
    cv.imshow(widnowName, frame) 
    if cv.waitKey(1) & 0xff == 27:
        cv.destroyAllWindows()


def main():
    captureFrame()


if __name__ == '__main__':
    main()

