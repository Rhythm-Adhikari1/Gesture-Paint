import cv2
import mediapipe as mp
import time
import math

class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):

        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param modelComplexity: Complexity of the hand landmark model: 0 or 1.
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, img_to_draw ,draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img_to_draw, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)

        return allHands, img_to_draw

    def fingersUp(self, myHand):
        """
        Determines which fingers are open (returns 1) or closed (returns 0) in a robust manner.
        This method rotates the hand so it appears upright and then compares landmark positions.
        
        :param myHand: Dictionary containing hand info:
                    - "type": "Right" or "Left"
                    - "lmList": List of 21 landmarks (each a [x, y, z] list)
        :return: List of 5 integers for [Thumb, Index, Middle, Ring, Pinky],
                where 1 means the finger is open and 0 means closed.
        """
        fingers = []
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        
        # Check that there are enough landmarks
        if not myLmList or len(myLmList) < 21:
            return [0, 0, 0, 0, 0]
        m_lmList = myLmList.copy()
        for i in range(len(m_lmList)):
            m_lmList[i][1] = -m_lmList[i][1]
        
        # Determine hand orientation using wrist (landmark 0) and middle finger MCP (landmark 9)
        wx, wy, _ = m_lmList[0]
        mx, my, _ = m_lmList[9]
        angle = math.atan2(my - wy, mx - wx)  # angle relative to horizontal
        rot_angle = -math.pi/2 -angle                   # rotation needed to make the hand upright

        def rotate_point(p):
            x, y, z = p
            # Translate the point so that the wrist becomes the origin.
            # Rotate the point by rot_angle using the 2D rotation formula.

            x_rot = wx + (x - wx) * math.cos(rot_angle) - (y - wy) * math.sin(rot_angle) 
            y_rot = wy + (x - wx) * math.sin(rot_angle) + (y - wy) * math.cos(rot_angle)
            return (x_rot, y_rot)
        
        # Create a list of rotated landmarks.
        rotated_points = [rotate_point(p) for p in m_lmList]
        
        # --- Thumb ---
        # Compare the x-coordinates after rotation.
        # For a right hand, if the thumb tip (landmark at self.tipIds[0]) is to the right of its preceding joint,
        # the thumb is open. Reverse the condition for a left hand.
        thumb_tip = rotated_points[self.tipIds[0]]
        thumb_ip  = rotated_points[self.tipIds[0] - 1]
        if myHandType == "Right":
            fingers.append(1 if thumb_tip[0] < thumb_ip[0] else 0)
        else:
            fingers.append(1 if thumb_tip[0] > thumb_ip[0] else 0)
        
        # --- Other Fingers (Index, Middle, Ring, Pinky) ---
        # For each of these fingers, if the fingertip (after rotation) is higher up 
        # (i.e. has a smaller y-coordinate) than a lower joint (typically landmark index: tipId - 2),
        # then the finger is considered open.
        for i in range(1, 5):
            tip = rotated_points[self.tipIds[i]]
            lower_joint = rotated_points[self.tipIds[i] - 2]
            fingers.append(0 if tip[1] > lower_joint[1] else 1)

        for i in range(len(m_lmList)):
            m_lmList[i][1] = -m_lmList[i][1]
        
        return fingers

    
    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Find the distance between two landmarks input should be (x1,y1) (x2,y2)
        :param p1: Point1 (x1,y1)
        :param p2: Point2 (x2,y2)
        :param img: Image to draw output on. If no image input output img is None
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, info, img

    

def main():
    # Initialize the webcam to capture video
    # The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
    cap = cv2.VideoCapture(0)

    # Initialize the HandDetector class with the given parameters
    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()

        # Find hands in the current frame
        # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
        # The 'flipType' parameter flips the image, making it easier for some detections
        hands, img = detector.findHands(img,img_to_draw= img, draw=True, flipType=True)

        # Check if any hands are detected
        if hands:
            # Information for the first hand detected
            hand1 = hands[0]  # Get the first hand detected
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

            # Count the number of fingers up for the first hand
            fingers1 = detector.fingersUp(hand1)
            print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

            # Calculate distance between specific landmarks on the first hand and draw it on the image
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),
                                                      scale=10)

            # Check if a second hand is detected
            if len(hands) == 2:
                # Information for the second hand
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                center2 = hand2['center']
                handType2 = hand2["type"]

                # Count the number of fingers up for the second hand
                fingers2 = detector.fingersUp(hand2)
                print(f'H2 = {fingers2.count(1)}', end=" ")

                # Calculate distance between the index fingers of both hands and draw it on the image
                length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                                                          scale=10)

            print(" ")  # New line for better readability of the printed output

        # Display the image in a window
        cv2.imshow("Image", img)

        # Keep the window open and update it for each frame; wait for 1 millisecond between frames
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
