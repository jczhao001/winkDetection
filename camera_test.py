import cv2
import pykinect2
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    cv2.imshow('img', img)
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()