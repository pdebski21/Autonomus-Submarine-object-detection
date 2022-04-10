import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def captureFrame():
    cap = cv.VideoCapture(0)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = detectCorners(frame)
        cv.imshow("window-name", frame)
        # cv.imwrite("frame%d.jpg" % count, frame)
        count = count + 1
        if cv.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()


def detectCorners(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    frame[dst > 0.01 * dst.max()] = [0, 255, 0]
    return frame


def FeatureMatching(template, frame, minMatchCnt=10):
    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(frame, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > minMatchCnt:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(
            -1, 1, 2
        )
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        w, h, c = template.shape
        pts = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
        ).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        frame = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        # better use for logging
        print(
            "Not enough matches are found - {}/{}".format(
                len(good), minMatchCount
            )
        )
        matchesMask = None

    draw_params = dict(
        matchColor=(0, 255, 0),  
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=2,
    )
    frame_matches = cv.drawMatches(
        template, kp1, frame, kp2, good, None, **draw_params
    )
    displayFrame(frame_matches, '3')
    

def histogramMatching(template, frame):
    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(template, frame, multichannel=multi)
    return matched




def compareHistograms(template, frame):
    template_hist = cv.calcHist([template],[0],None,[256],[0,256])
    frame_hist = cv.calcHist([template],[0],None,[256],[0,256])
    color = ('b','g','r')
    for i,col in enumerate(color):
        template_histr = cv.calcHist([template],[i],None,[256],[0,256])
        frame_histr = cv.calcHist([frame],[i],None,[256],[0,256])
        plt.plot(template_histr,color = col)
        plt.plot(frame_histr,color = col)
        plt.xlim([0,256])
    plt.show()


def displayFrame(frame, windowName):
    cv.imshow(windowName, frame)
    if cv.waitKey(0) & 0xFF == ord('q'):
        cv.destroyAllWindows()


def main():
    # captureFrame()
    template = cv.imread(
        "/home/piotrdebski/Pulpit/Robocik/board_detection/samples/board_images/object_gangster.png"
    )
    frame = cv.imread(
        "/home/piotrdebski/Pulpit/Robocik/board_detection/samples/board_images/objects.png"
    )
    FeatureMatching(template, frame)
    template_corners = detectCorners(template)
    frame_corners = detectCorners(frame)
    displayFrame(template_corners, "1")
    displayFrame(frame_corners, "2")


if __name__ == "__main__":
    main()
