import dlib
import cv2
import numpy as np

def empty(a):
    pass

def createbox(img,points,masked=True,croped=False):
  if croped:
    x,y,w,h=cv2.boundingRect(points)
    imgcroped=img[y:y+h,x:x+w]
    return imgcroped
  if masked:
    mask=np.zeros_like(img)
    mask=cv2.fillPoly(mask,[points],(255,255,255))
    return mask

vid=cv2.VideoCapture(0)
win='Color My Lips'
cv2.namedWindow(win)
cv2.createTrackbar('Red', win, 1, 255, empty)
cv2.createTrackbar('Green', win, 1, 255, empty)
cv2.createTrackbar('Blue', win, 1, 255, empty)

while True:
    stat,frame=vid.read()
    r=cv2.getTrackbarPos('Red', win)
    g= cv2.getTrackbarPos('Green', win)
    b= cv2.getTrackbarPos('Blue', win)
    frame_colored=frame.copy()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face = detector(frame)
    for f in face:
        x1, y1 = f.left(), f.top()
        x2, y2 = f.right(), f.bottom()
        landmarks = predictor(frame, f)
        points = []
        for n in range(68):
            i = landmarks.part(n).x
            j = landmarks.part(n).y
            points.append([i, j])
            cv2.circle(frame, (i, j), 4, (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, str(n), (i, j + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
    points = np.array(points)
    img_lip = createbox(frame_colored, points[48:61])
    lip_color = np.zeros_like(img_lip)
    lip_color[:]=b,g,r
    colored_lip = cv2.bitwise_and(img_lip, lip_color)
    colored_lip = cv2.GaussianBlur(colored_lip, (7, 7), 10)
    frame_colored = cv2.addWeighted(frame_colored, 1, colored_lip, 0.4, 0)
    cv2.imshow(win,frame_colored)
    cv2.waitKey(1)
