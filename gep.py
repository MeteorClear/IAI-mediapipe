import cv2
import math
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


# float
def cal_distance(x1,x2,y1,y2,z1=0,z2=0):
  return ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2) ** (1/2)

# int
def cal_mid_point(x1,x2,y1,y2,z1=0,z2=0):
  if z1==0 and z2==0:
    return ((x1+x2)//2, (y1+y2)//2)
  else:
    return ((x1+x2)//2, (y1+y2)//2, (z1+z2)//2)

# radian
def cal_three_point_angle(x1,x2,y1,y2,midptx,midpty):
  return (math.atan((y2-midpty)/(x2-midptx)) - math.atan((y1-midpty)/(x1-midptx)))

#atan( (점1 Y - 점2 Y) / ( 점1 X- 점2X)) - atan( (점 3 Y - 점2Y) / (점3 X - 점2X))  * 180/pi


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


    angle_image = np.zeros(image.shape, dtype=np.uint8)
    angle_image[:] = (160, 160, 160)
    adi_height, adi_width, adi_c = angle_image.shape
    cv2.line(angle_image, (adi_width//2-1, 0), (adi_width//2-1, adi_height), (255,255,255), 2)
    cv2.line(angle_image, (0, adi_height//2-1), (adi_width, adi_height//2-1), (255,255,255), 2)
    #print(adi_width//2,adi_height//2)
    
    


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

        # in for loop
      ######################### 
      # write

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

##########################################################################
    # write
    
    temp_midpoint = -1
    t_distance = -1



########################################################################## gray_image section1

    try:
      temp_midpoint = cal_mid_point(landmarks_coordinates[4][0],landmarks_coordinates[8][0],landmarks_coordinates[4][1],landmarks_coordinates[8][1])
      #print(temp_midpoint)
    except:
      pass

    try:
      cv2.circle(gray_image, (landmarks_coordinates[8][0],landmarks_coordinates[8][1]), 12, (255,255,255), 2)
      cv2.line(gray_image, (landmarks_coordinates[8][0],landmarks_coordinates[8][1]), (landmarks_coordinates[4][0],landmarks_coordinates[4][1]), (255,255,255), 2)
      cv2.circle(gray_image, temp_midpoint, 5, (0,255,0), 2)
      
    except:
      pass


########################################################################## gray_image section2
    gray_image = cv2.flip(gray_image, 1)

    
    try:
      cv2.putText(gray_image, "("+str(landmarks_coordinates[8][0])+","+str(landmarks_coordinates[8][1])+")", (20,50), cv2.FONT_ITALIC, 1, (255,0,0), 2)
      t_midx, t_midy = temp_midpoint
      t_distance = cal_distance(landmarks_coordinates[4][0],landmarks_coordinates[8][0],landmarks_coordinates[4][1],landmarks_coordinates[8][1])
      t_distance = round(t_distance, 3)
      cv2.putText(gray_image, str(t_distance), (image_width-t_midx-30, t_midy-15), cv2.FONT_ITALIC, 0.5, (0,0,255), 2)

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



########################################################################## angle_image section1

    img_midpt_x, img_midpt_y = image_width//2, image_height//2
    anglex, angley = -1, -1

    try:
      act_midx, act_midy = temp_midpoint
      
      nor_x, nor_y = abs(img_midpt_x-act_midx), abs(img_midpt_y-act_midy)

      if act_midx>img_midpt_x:
        nor_x = -nor_x
      if act_midy>img_midpt_y:
        nor_y = -nor_y

      #print(act_midx+nor_x, act_midy+nor_y)
      #print(image_width,img_midpt_y,nor_y,act_midy,act_midy+nor_y)
      cv2.line(angle_image, (landmarks_coordinates[4][0]+nor_x, landmarks_coordinates[4][1]+nor_y), (landmarks_coordinates[8][0]+nor_x, landmarks_coordinates[8][1]+nor_y), (50,255,50), 2)
      cv2.circle(angle_image, (act_midx+nor_x, act_midy+nor_y), 5, (0,0,255), 2)
      cv2.circle(angle_image, (landmarks_coordinates[4][0]+nor_x, landmarks_coordinates[4][1]+nor_y), 5, (255,0,0), 2)
      cv2.circle(angle_image, (landmarks_coordinates[8][0]+nor_x, landmarks_coordinates[8][1]+nor_y), 5, (255,0,0), 2)
      
      anglex, angley = landmarks_coordinates[8][0]+nor_x, landmarks_coordinates[8][1]+nor_y
    except:
      pass



########################################################################## angle_image section2
    angle_image = cv2.flip(angle_image, 1)



    try:
      t_text = str(t_distance)
      textsize = cv2.getTextSize(t_text, cv2.FONT_ITALIC, 1, 2)[0]
      textX = (angle_image.shape[1] - textsize[0]) // 2
      textY = (angle_image.shape[0] + textsize[1]) // 2

      t_midx, t_midy = temp_midpoint
      cv2.putText(angle_image, str(t_distance), (textX+10, textY+150), cv2.FONT_ITALIC, 0.8, (0,0,255), 2)

    except:
      cv2.putText(angle_image, "-Undetected-", (adi_width//2-85, textY+150), cv2.FONT_ITALIC, 0.8, (255,0,0), 2)


    try:
      t_angle = cal_three_point_angle(anglex, img_midpt_x-1, angley, 1, img_midpt_x, img_midpt_y)
      print(t_angle,(math.pi/180)*t_angle)
      
    except:
      pass
    




##########################################################################
    
    cv2.imshow('sub', gray_image)
    cv2.imshow('debug', angle_image)

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