# Detect Face From An Image
# author: Tonoy<tonoy.sust@gmail.com>

import cv2


def main():
    cap = cv2.VideoCapture(0)

    cascade = cv2.CascadeClassifier(
        '../assets/haarcascade_frontalface_default.xml')

    while cap.isOpened():
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_white = cv2.equalizeHist(gray)

        rects = cascade.detectMultiScale(
            black_white, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for x, y, w, h in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
