import cv2
import numpy as np
import mediapipe as mp
import pyautogui as pg

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

SYSMODE = 1





key_check = { 'up':False, 'down':False, 'left':False, 'right':False,
            'pageup':False, 'pagedown':False }


def key_turn_on(ikey):
  global key_check
  if not key_check[ikey]:
    pg.keyDown(ikey)
    key_check[ikey] = True

def key_turn_off(ikey):
  global key_check
  if key_check[ikey]:
    pg.keyUp(ikey)
    key_check[ikey] = False



def check_position_move(pos,cw,ch):
  global key_check

  if SYSMODE != 1:
    return False
  
  try:
    nw, nh = pos
  except:
    return

  if nh < ch*0.25:
    key_turn_off('down')
    key_turn_on('up')
  elif nh > ch-ch*0.25:
    key_turn_off('up')
    key_turn_on('down')
  else:
    key_turn_off('up')
    key_turn_off('down')

  if nw > cw-cw*0.25:
    key_turn_off('right')
    key_turn_on('left')
  elif nw < cw*0.25:
    key_turn_off('left')
    key_turn_on('right')
  else:
    key_turn_off('left')
    key_turn_off('right')
    





# 확대 축소 페이지업 다운
def check_zoom_rotate(pos,cw,ch):
  global key_check

  if SYSMODE != 2:
    return False
  
  try:
    nw, nh = pos
  except:
    return

  if nh < ch*0.25:
    key_turn_off('pagedown')
    key_turn_on('pageup')
  elif nh > ch-ch*0.25:
    key_turn_off('pageup')
    key_turn_on('pagedown')
  else:
    key_turn_off('pageup')
    key_turn_off('pagedown')




# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9) as hands:

  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    
    
    image_height, image_width, channel = image.shape
    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    gray_image = np.zeros(image.shape, dtype=np.uint8)
    gray_image[:] = (160, 160, 160)
    gi_height, gi_width, gi_c = gray_image.shape
    cv2.rectangle(gray_image, (0, 0), (gi_width, int(gi_height*0.25)), (255,255,255), 2)
    cv2.rectangle(gray_image, (gi_width, gi_height), (0, int(gi_height-(gi_height*0.25))), (255,255,255), 2)
    cv2.rectangle(gray_image, (0, 0), (int(gi_width*0.25), gi_height), (255,255,255), 2)
    cv2.rectangle(gray_image, (gi_width, gi_height), (int(gi_width-(gi_width*0.25)), 0), (255,255,255), 2)
    


    low_landmarks_coordinates = []
    landmarks_coordinates = []
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        for idx, landmark_cor in enumerate(hand_landmarks.landmark):
          landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark_cor.x, landmark_cor.y,
                                                                    image_width, image_height)

          low_landmarks_coordinates.append([landmark_cor.x,landmark_cor.y,landmark_cor.z])
          landmarks_coordinates.append(landmark_px)

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        mp_drawing.draw_landmarks(
            gray_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

      if SYSMODE==0:
        pass
      elif SYSMODE==1:
        check_position_move(landmarks_coordinates[8], image_width, image_height)
      elif SYSMODE==2:
        check_zoom_rotate(landmarks_coordinates[8], image_width, image_height)
      else:
        print("ERROR : UNKOWN MODE")
        print(SYSMODE)
        break
    
     
    cv2.imshow('main frame', cv2.flip(image, 1))

    try:
      cv2.circle(gray_image, (landmarks_coordinates[8][0],landmarks_coordinates[8][1]), 12, (255,255,2550), 2)
    except:
      pass

    gray_image = cv2.flip(gray_image, 1)

    
    
    try:
      cv2.putText(gray_image, "("+str(landmarks_coordinates[8][0])+","+str(landmarks_coordinates[8][1])+")", (20,50), cv2.FONT_ITALIC, 1, (255,0,0), 2)
    except:
      cv2.putText(gray_image, "-Undetected-", (20,50), cv2.FONT_ITALIC, 1, (255,0,0), 2)

    if SYSMODE==1:
      try:
        if key_check['up']:
          cv2.putText(gray_image, "UP", (20,70), cv2.FONT_ITALIC, 0.5, (0,255,0), 2)
        else:
          cv2.putText(gray_image, "UP", (20,70), cv2.FONT_ITALIC, 0.5, (0,0,255), 2)

        if key_check['down']:
          cv2.putText(gray_image, "DOWN", (20,85), cv2.FONT_ITALIC, 0.5, (0,255,0), 2)
        else:
          cv2.putText(gray_image, "DOWN", (20,85), cv2.FONT_ITALIC, 0.5, (0,0,255), 2)

        if key_check['left']:
          cv2.putText(gray_image, "LEFT", (20,100), cv2.FONT_ITALIC, 0.5, (0,255,0), 2)
        else:
          cv2.putText(gray_image, "LEFT", (20,100), cv2.FONT_ITALIC, 0.5, (0,0,255), 2)

        if key_check['right']:
          cv2.putText(gray_image, "RIGHT", (20,115), cv2.FONT_ITALIC, 0.5, (0,255,0), 2)
        else:
          cv2.putText(gray_image, "RIGHT", (20,115), cv2.FONT_ITALIC, 0.5, (0,0,255), 2)
      except:
        pass
    
    
    cv2.imshow('sub', gray_image)

    key_input = cv2.waitKey(5)
    
    
    if key_input & 0xFF == 27:
      break
    elif key_input & 0xFF == 32:
      SYSMODE = (SYSMODE+1) % 3
      print(SYSMODE)
    else:
      if key_input & 0xFF == 255 or key_input & 0xFF == 0:
        pass
      else:
        print(key_input & 0xFF)

cap.release()