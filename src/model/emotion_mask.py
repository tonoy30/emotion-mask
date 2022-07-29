"""Test for face detection"""

from os.path import realpath

import cv2

from masking import apply_mask
from model import get_image_to_emotion_predictor


def main():

    cap = cv2.VideoCapture(0)
    # load mask
    mask0 = cv2.imread(realpath('src/assets/dog.png'))
    mask1 = cv2.imread(realpath('src/assets/dalmation.png'))
    mask2 = cv2.imread(realpath('src/assets/sheepdog.png'))
    masks = (mask0, mask1, mask2)

    # get emotion predictor
    predictor = get_image_to_emotion_predictor(
        realpath('src/assets/model_best.pth'))

    # initialize front face classifier
    cascade = cv2.CascadeClassifier(
        realpath('src/assets/haarcascade_frontalface_default.xml'))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_h, frame_w, _ = frame.shape

        # Convert to black-and-white
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blackwhite = cv2.equalizeHist(gray)

        rects = cascade.detectMultiScale(
            blackwhite, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for x, y, w, h in rects:
            # crop a frame slightly larger than the face
            y0, y1 = int(y - 0.25*h), int(y + 0.75*h)
            x0, x1 = x, x + w
            # give up if the cropped frame would be out-of-bounds
            if x0 < 0 or y0 < 0 or x1 > frame_w or y1 > frame_h:
                continue
            # apply mask
            mask = masks[predictor(frame[y:y+h, x: x+w])]
            frame[y0: y1, x0: x1] = apply_mask(frame[y0: y1, x0: x1], mask)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
