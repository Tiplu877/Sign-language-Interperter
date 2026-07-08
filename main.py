import cv2
import mediapipe as mp
import csv
import pickle
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()
model = pickle.load(open('model.pkl', 'rb'))
while(True):
    landmark_list = []
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        wrist_x = hand_landmarks.landmark[0].x
        palm_x = hand_landmarks.landmark[9].x
        distance = abs(wrist_x - palm_x)
        for i in range(21):
                lm_x = hand_landmarks.landmark[i].x
                lm_y = hand_landmarks.landmark[i].y
                norm_x = (lm_x-wrist_x)/distance
                norm_y = (lm_y-hand_landmarks.landmark[0].y)/distance
                landmark_list.append((norm_x, norm_y))
    key = chr(cv2.waitKey(1) & 0xFF)
    if(key.isascii() and key.isalpha() and len(landmark_list)>0):
        with open('data.csv','a') as f:
            writer = csv.writer(f)
            flat_list = [val for lm in landmark_list for val in lm]
            writer.writerow([key] + flat_list)
    if(len(landmark_list)>0):
        flat_list = [val for lm in landmark_list for val in lm]
        prediction = model.predict([flat_list])
        cv2.rectangle(frame, (0,0), (200,130), (0,0,0), -1)
        cv2.putText(frame, prediction[0], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255),2)
    cv2.imshow('frame', frame)

   
    if key == ('q'):
        break
