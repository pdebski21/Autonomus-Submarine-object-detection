import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import exposure

def captureFrame(filename=0):
    cap = cv.VideoCapture(filename)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        detectColor(frame)
        frame = detectCorners(frame)
        cv.imshow("capture", frame)
        # cv.imwrite("frame%d.jpg" % count, frame)
        count = count + 1
        if cv.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()

def detectColor(frame):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
   
    # White color
    low_white = np.array([84,200,100])
    high_white = np.array([93,255,144])
    white_mask = cv.inRange(hsv_frame, low_white, high_white)
    white = cv.bitwise_and(frame, frame, mask=white_mask)
    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([172, 255, 255])
    red_mask = cv.inRange(hsv_frame, low_red, high_red)
    red = cv.bitwise_and(frame, frame, mask=red_mask)
    # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv.inRange(hsv_frame, low_blue, high_blue)
    blue = cv.bitwise_and(frame, frame, mask=blue_mask)
    # Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv.inRange(hsv_frame, low_green, high_green)
    green = cv.bitwise_and(frame, frame, mask=green_mask)
    # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv.inRange(hsv_frame, low, high)
    result = cv.bitwise_and(frame, frame, mask=mask)
 
    cv.imshow("White", white)
    cv.imshow("Red", red)
    cv.imshow("Blue", blue)
    cv.imshow("Green", green)
    cv.imshow("Result", result)

    # corners detection    
    corners = detectCorners(white)
    cv.imshow("Corners", corners)

    # erode with cross kernel
    eroded = erosion(white, 3, 2)
    cv.imshow("Eroded", eroded)

    # dilate with rect kernel
    dilatated = dilatation(eroded, 5, 0)    
    cv.imshow("Dilatated", dilatated)
    

def detectCorners(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    #dst = cv.dilate(dst, None)
    frame[dst > 0.01 * dst.max()] = [0, 255, 0]
    return frame


def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE


def erosion(frame, erosion_size, erosion_shape):
    erosion_shape = morph_shape(erosion_shape)
    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    erosion_res = cv.erode(frame, element)
    return erosion_res


def dilatation(frame, dilatation_size, dilatation_shape):
    dilatation_shape = morph_shape(dilatation_shape)
    element = cv.getStructuringElement(dilatation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_res = cv.dilate(frame, element)
    return dilatation_res

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
        #frame = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        # better use for logging
        print(
            "Not enough matches are found - {}/{}".format(
                len(good), minMatchCnt
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
    

def matchHistograms(template, frame):
    multi = True if template.shape[-1] > 1 and frame.shape[-1] > 1 else False
    matched = exposure.match_histograms(template, frame, multichannel=multi)
    final_frame = cv.hconcat((template, frame, matched))
    #displayFrame(matched, 'matched_histograms')
    return matched


def compareHistograms(template, frame):
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[10:750, 10:370] = 255
    masked_frame = cv.bitwise_and(frame,frame,mask = mask)
    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv.calcHist([frame],[0],None,[256],[0,256])
    hist_mask = cv.calcHist([frame],[0],mask,[256],[0,256])
    plt.subplot(221), plt.imshow(frame, 'gray')
    plt.subplot(222), plt.imshow(mask,'gray')
    plt.subplot(223), plt.imshow(masked_frame, 'gray')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    plt.xlim([0,256])
    plt.show()


def displayFrame(frame, windowName):
    cv.resizeWindow(windowName, 300, 400)
    cv.imshow(windowName, frame)
    if cv.waitKey(0) & 0xFF == ord('q'):
        cv.destroyAllWindows()


def main():
    captureFrame("samples/board_video/2016_0922_050010_016.MP4")
    """
    captureFrame()
    template = cv.imread(
        "/home/piotrdebski/Pulpit/Robocik/board_detection/samples/board_images/object_gangster.png"
    )
    frame = cv.imread(
        "/home/piotrdebski/Pulpit/Robocik/board_detection/samples/board_images/objects.png"
        #"/home/piotrdebski/Pulpit/Robocik/board_detection/samples/board_images/pedestraints.png"
    )
    frame_grey = cv.imread(
        "/home/piotrdebski/Pulpit/Robocik/board_detection/samples/board_images/objects_grey.png"
    ) 

 
    FeatureMatching(template, frame)
    template_corners = detectCorners(template)
    frame_corners = detectCorners(frame)
    displayFrame(template_corners, "1")
    displayFrame(frame_corners, "2")
    """
    #compareHistograms(template, frame)
    #matched = matchHistograms(frame, frame_grey)

if __name__ == "__main__":
    main()
