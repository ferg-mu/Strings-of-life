import cv2
import mediapipe as mp
import math
import numpy as np
import time
import keyboard
from pythonosc import udp_client
from pythonosc import osc_message_builder


# min & max values for normalisation
# area should aim to be 
MAX_AREA = 40000  # this should be set to a plausible maximum area
MAX_LENGTH = 200  # this should be set to a plausible maximum length for any line between fingers
INDEX_MAX_LENGTH = 200
PINKY_MAX_LENGTH = 250
ip = "192.168.1.138"
port = 55565
client = udp_client.SimpleUDPClient(ip, port)
index = 0


class HandProcessor:
    def __init__(self, landmarks, frame_shape, handedness, client):
        self.landmarks = landmarks
        self.frame_shape = frame_shape
        self.handedness = handedness
        self.client = client
        self.thumb_tip_pixel = (0, 0)
        self.index_finger_tip_pixel = (0, 0)
        self.pinky_tip_pixel = (0, 0)
        self.middle_finger_tip_pixel = (0, 0)
        self.ring_finger_tip_pixel = (0, 0)
        self.wrist_tip_pixel = (0, 0)
        self.index_line_length = 0
        self.middle_line_length = 0
        self.pinky_line_length = 0
        self.ring_line_length = 0
        self.max_circle = None
        self.palm_circle = None
        self.area_values = []
        self.x_pos_0_values = []
        self.x_pos_1_values = []
        self.y_pos_0_values = []
        self.y_pos_1_values = []
        self.index_length_values = []
        self.pinky_length_values = []
        self.calculate_metrics()  # Ensuring this is called after all initializations


    def calculate_metrics(self):
        thumb_tip = self.landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_finger_tip = self.landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = self.landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
        middle_finger_tip = self.landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = self.landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        wrist = self.landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
        index_finger_mpc = self.landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mpc = self.landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]

        # Convert normalized coordinates to pixel coordinates as NumPy arrays
        self.thumb_tip_pixel = np.array([thumb_tip.x, thumb_tip.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.index_finger_tip_pixel = np.array([index_finger_tip.x, index_finger_tip.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.pinky_tip_pixel = np.array([pinky_tip.x, pinky_tip.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.middle_finger_tip_pixel = np.array([middle_finger_tip.x, middle_finger_tip.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.ring_finger_tip_pixel = np.array([ring_finger_tip.x, ring_finger_tip.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.wrist_tip_pixel = np.array([wrist.x, wrist.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.index_finger_mpc_pixel = np.array([index_finger_mpc.x, index_finger_mpc.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])
        self.pinky_mpc_pixel = np.array([pinky_mpc.x, pinky_mpc.y]) * np.array([self.frame_shape[1], self.frame_shape[0]])

        points = np.array([[lm.x * self.frame_shape[1], lm.y * self.frame_shape[0]] for lm in self.landmarks.landmark]).astype(int)
        center, radius = cv2.minEnclosingCircle(points)
        # print("the centre point is", center)
        self.max_circle = (center, radius)
        self.area = math.pi * radius ** 2

        delta_x = center[0] - self.wrist_tip_pixel[0]
        delta_y = center[1] - self.wrist_tip_pixel[1]
        angle_from_horizontal = math.degrees(math.atan2(delta_y, delta_x))
        self.wrist_to_center_angle = (angle_from_horizontal + 90) % 360

        # Calculate the enclosing circle for the palm
        palm_points = np.array([self.index_finger_mpc_pixel, self.pinky_mpc_pixel, self.wrist_tip_pixel]).astype(int)
        palm_center, palm_radius = cv2.minEnclosingCircle(palm_points)
        self.palm_circle = (palm_center, palm_radius)

        # Calculate distances using NumPy for element-wise subtraction
        self.index_line_length = np.linalg.norm(self.index_finger_tip_pixel - self.thumb_tip_pixel)
        self.pinky_line_length = np.linalg.norm(self.pinky_tip_pixel - self.thumb_tip_pixel)
        self.middle_line_length = np.linalg.norm(self.middle_finger_tip_pixel - self.thumb_tip_pixel)
        self.ring_line_length = np.linalg.norm(self.ring_finger_tip_pixel - self.thumb_tip_pixel)


        # define the robot zone for index 0
        range_0_axis_x = [0.25, 0.5]
        range_1_axis_x = [0.6, 0.75]
        range_0_axis_y = [0, 0.2]
        range_1_axis_y = [1.0, 0.8]
        range_axis_z = [0.55, 0.75]
        range_index_thumb = [0.48, 0.57]
        range_pinky_thumb = [0.4, 0.6]

        area_min = 20000
        area_max = 125000

        # if the there are 2 hands in the image control will be given to the hand with the largest area, the smaller hand will be ignored
        # if the area of the hand is less than 30000 the hand will be ignored
        if self.area < 30000:
            return
        
        index_min = 20
        index_max = 200
        # index_max = [200, 450]
        
        pinky_min = 20
        pinky_max = 225
        # pinky_max = [250, 350]

        # pos normalisation
        self.x_pos_0 = HandProcessor.remap_value(center[0], 0, self.frame_shape[1], range_0_axis_x[0], range_0_axis_x[1])
        self.x_pos_1 = HandProcessor.remap_value(center[0], 0, self.frame_shape[1], range_1_axis_x[0], range_1_axis_x[1])
        
        self.z_pos = HandProcessor.remap_value(center[1], 0, self.frame_shape[0], range_axis_z[1], range_axis_z[0])
        
        self.y_pos_0 = HandProcessor.remap_value(self.area, area_min, area_max, range_0_axis_y[1], range_0_axis_y[0])
        # print("Y t1: ", self.y_pos_0)
        self.y_pos_0 = min(max(self.y_pos_0, range_0_axis_y[0]), range_0_axis_y[1])
        # print("Y t2: ",self.y_pos_0)
 
        
        self.y_pos_1 = HandProcessor.remap_value(self.area, area_min, area_max, range_1_axis_y[1], range_1_axis_y[0])
        self.y_pos_1 = max(min(self.y_pos_1, range_1_axis_y[0]), range_1_axis_y[1])
        # index normalisation
        self.index_line_length = HandProcessor.remap_value(self.index_line_length, index_min, index_max, range_index_thumb[0], range_index_thumb[1])
        self.index_line_length = max(min(self.index_line_length, range_index_thumb[1]), range_index_thumb[0])

        # pinky normalisation
        self.pinky_line_length = HandProcessor.remap_value(self.pinky_line_length, pinky_min, pinky_max, range_pinky_thumb[0], range_pinky_thumb[1])
        self.pinky_line_length = max(min(self.pinky_line_length, range_pinky_thumb[1]), range_pinky_thumb[0])

        # de-noising values
        self.area_values.append(self.area)
        self.x_pos_0_values.append(self.x_pos_0)
        self.x_pos_1_values.append(self.x_pos_1)
        self.y_pos_0_values.append(self.y_pos_0)
        self.y_pos_1_values.append(self.y_pos_1)
        self.index_length_values.append(self.index_line_length)
        self.pinky_length_values.append(self.pinky_line_length)

        # Process each list like the area values
        if len(self.area_values) == 10:
            self.area = sum(self.area_values) / len(self.area_values)
            self.x_pos_0 = sum(self.x_pos_0_values) / len(self.x_pos_0_values)
            self.x_pos_1 = sum(self.x_pos_1_values) / len(self.x_pos_1_values)
            self.y_pos_0 = sum(self.y_pos_0_values) / len(self.y_pos_0_values)
            self.y_pos_1 = sum(self.y_pos_1_values) / len(self.y_pos_1_values)
            self.index_line_length = sum(self.index_length_values) / len(self.index_length_values)
            self.pinky_line_length = sum(self.pinky_length_values) / len(self.pinky_length_values)
            
            
        
            
            # Clear lists for next set of data
            self.area_values.clear()
            self.x_pos_0_values.clear()
            self.x_pos_1_values.clear()
            self.y_pos_0_values.clear()
            self.y_pos_1_values.clear()
            self.index_length_values.clear()
            self.pinky_length_values.clear()
            
        # print to terminal
        print(" X0: ", round(self.x_pos_0, 3), " X1: ", round(self.x_pos_1, 3), "Y0: ", round(self.y_pos_0, 3),"Y1: ",round(self.y_pos_1, 3),"Z: ", round(self.z_pos, 3), " Index: ", round(self.index_line_length, 2), " Pinky: ", round(self.pinky_line_length, 2), "palm radius: ", round(palm_radius, 2))
            

        if index == 1:

            address = "/motion/rot"
            rx = -1.0 * self.pinky_line_length
            ry = -1.0 - self.index_line_length
            rz = 0
            message = [index, rx, ry, rz]
            client.send_message(address, message)
            
            # y_pos to control z
            # x_pos to control x
            # area to control y
            address = "/motion/pos"
            x = 1.0-self.x_pos_1
            y = 1.0-self.y_pos_1
            z = self.z_pos
            message = [index, x, y, z]
            # print(message)
            client.send_message(address, message)

        elif index == 0:
            address = "/motion/rot"
            rx = -1.0 * self.pinky_line_length + 0.5
            ry = self.index_line_length + 0.5
            rz = 0#self.pinky_line_length
            message = [index, rx, ry, rz]
            client.send_message(address, message)
            
            # y_pos to control z
            # x_pos to control x
            # area to control y
            address = "/motion/pos"
            x = 1.0-self.x_pos_0
            y = self.y_pos_0
            z = self.z_pos
            message = [index, x, y, z]
            # print(message)
            client.send_message(address, message)


            # testing logic of ROI split image workflow
            # workflow should be able to detect if a hand is on the left or right ROI
            # if a hand is detected in the left ROI it will send control to index 1
            # if a hand is detected in the right ROI it will send control to index 0
            # if one hand is detected in the left ROI and the other hand is detected in the right ROI, the left ROI will take control of index 1 and the right ROI will take control of index 0
            # if no hand is detected in the left or right ROI, the robot will be in a neutral position
        

    def draw_hand(self, image):
        if self.max_circle:
            cv2.circle(image, (int(self.max_circle[0][0]), int(self.max_circle[0][1])), int(self.max_circle[1]), (255, 255, 255), 3)

        if self.palm_circle:
            cv2.circle(image, (int(self.palm_circle[0][0]), int(self.palm_circle[0][1])), int(self.palm_circle[1]), (255, 255, 255), 3)

        # Additional drawing logic
        mp.solutions.drawing_utils.draw_landmarks(
            image, self.landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=1),
            mp.solutions.drawing_utils.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=1)
        )
        # Draw hand skeleton
        mp.solutions.drawing_utils.draw_landmarks(
            image, self.landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=1),
            mp.solutions.drawing_utils.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=1)
            
            # presentation style
            # mp.solutions.drawing_utils.DrawingSpec(color=(75, 75, 75), thickness=2, circle_radius=1),
            # mp.solutions.drawing_utils.DrawingSpec(color=(75, 75, 75), thickness=2, circle_radius=2)
        )

        # Draw lines between fingers, ensure coordinates are integer tuples
        # cv2.line(image, tuple(self.thumb_tip_pixel.astype(int)), tuple(self.index_finger_tip_pixel.astype(int)), (255, 0, 0), 3)
        # cv2.line(image, tuple(self.thumb_tip_pixel.astype(int)), tuple(self.pinky_tip_pixel.astype(int)), (0, 255, 0), 3)
        # cv2.line(image, tuple(self.thumb_tip_pixel.astype(int)), tuple(self.middle_finger_tip_pixel.astype(int)), (0, 0, 255), 2)
        # cv2.line(image, tuple(self.thumb_tip_pixel.astype(int)), tuple(self.ring_finger_tip_pixel.astype(int)), (255, 0, 0), 2)

    @staticmethod
    def remap_value(value, old_min, old_max, new_min, new_max):
        return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)
client = udp_client.SimpleUDPClient(ip, port)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    black_image = np.zeros(frame.shape, dtype=np.uint8)
    # white_image = np.ones(frame.shape, dtype=np.uint8) *255

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            hand_processor = HandProcessor(hand_landmarks, frame.shape[:2], hand_label, client)
            hand_processor.draw_hand(black_image)
            # hand_processor.draw_hand(white_image)
            # hand_processor.print_metrics()

    # cv2.imshow('Hand Pose', black_image)
    cv2.imshow('Hand Pose', black_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()