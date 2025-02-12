import  cv2
import numpy as np
import math
from HandTrackingModule import  HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon= 0.8)



while True:
    success, img = cap.read()
    # hands, img = detector.findHands(img)

    
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, img_to_draw=img)    

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
