import cv2

video = cv2.VideoCapture(0)

while(video.isOpened()):
    ret,frame= video.read()
    if ret == True:
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video.release()
cv2.destroyAllWindows()