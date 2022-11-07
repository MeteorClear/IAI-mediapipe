import cv2
import math
import time
import numpy as np
import mediapipe as mp
import pyautogui as pg

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

SYSMODE = 1
FIND_CHECK = False

lock_check = False
lock_start = False
lock_end = False
lock_time_start = 0
lock_time_check = 0

LOCK_LIMIT_TIME = 3
LOCKED_position = []

ANGLE_RANGE = 30

IS_MOVE = False

prevTime = 0
curTime = 0



key_check = { 'up':False, 'down':False, 'left':False, 'right':False,
            'pageup':False, 'pagedown':False, 'shift':False }


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

  if not key_check['shift']:
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


def check_zoom_inout(l_d, n_d):
  global key_check

  if n_d > (l_d * (1 + 0.4)):
    key_turn_off('pagedown')
    key_turn_on('pageup')
  elif n_d < (l_d * (1 - 0.5)):
    key_turn_off('pageup')
    key_turn_on('pagedown')
  else:
    key_turn_off('pageup')
    key_turn_off('pagedown')


def check_rotate_move(lock_pos, now_pos, angle):
  global key_check
  if not key_check['shift']:
    if key_check['left'] or key_check['right'] or key_check['up'] or key_check['down']:
      return

  
  if 90+ANGLE_RANGE > angle > 90-ANGLE_RANGE:
    key_turn_on('shift')
    if lock_pos < now_pos:
      key_turn_off('right')
      key_turn_on('left')
    else: 
      key_turn_off('left')
      key_turn_on('right')
  else:
    key_turn_off('right')
    key_turn_off('shift')
    key_turn_off('left')

  



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



#두점 사이의 각도:(getAngle3P 계산용) 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
def angle_between(p1, p2):
  ang1 = np.arctan2(*p1[::-1])
  ang2 = np.arctan2(*p2[::-1])
  res = np.rad2deg((ang1 - ang2) % (2 * np.pi))
  return res
#세점 사이의 각도 1->2->3
def getAngle3P(p1, p2, p3, direction="CW"): 
  pt1 = (p1[0] - p2[0], p1[1] - p2[1])
  pt2 = (p3[0] - p2[0], p3[1] - p2[1])
  res = angle_between(pt1, pt2)
  res = (res + 360) % 360
  if direction == "CCW":    #반시계방향
    res = (360 - res) % 360
  return res











# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6) as hands:

  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = int(1/sec)



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
      FIND_CHECK = True
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

      pro_midpt = -1
      pro_degree = -1

      try:
        pro_midpt = cal_mid_point(landmarks_coordinates[4][0],landmarks_coordinates[8][0],landmarks_coordinates[4][1],landmarks_coordinates[8][1])
      except:
        pass
      
      '''
      try:
        pro_midx, pro_midy = pro_midpt
        pro_norx, pro_nory = abs(image_width//2-pro_midx), abs(image_height//2-pro_midy)

        if pro_midx > image_width//2:
          pro_norx = -pro_norx
        if pro_midy > image_height//2:
          pro_nory = -pro_nory

        t_l_angle = cal_three_point_angle(landmarks_coordinates[8][0]+pro_norx, l_anglex, landmarks_coordinates[8][1]+pro_nory, l_angley, image_width//2, image_height//2)
        t_l_angle_degree = (t_l_angle/math.pi*180)
        if t_l_angle_degree < 0:
          t_l_angle_degree += 180
        t_l_angle_degree = round(t_l_angle_degree, 3)
      except:
        pass
      '''

      if SYSMODE==0:
        pass
      elif SYSMODE==1:
        check_position_move(pro_midpt, image_width, image_height)
        if len(landmarks_coordinates) < 10:
          key_turn_off('up')
          key_turn_off('down')
          key_turn_off('left')
          key_turn_off('right')
      elif SYSMODE==2:
        #check_zoom_rotate(landmarks_coordinates[8], image_width, image_height)
        pass
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
    l_anglex, l_angley = -1, -1

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

    try:
      temp_l_midpoint = cal_mid_point(LOCKED_position[4][0],LOCKED_position[8][0],LOCKED_position[4][1],LOCKED_position[8][1])
      act_l_midx, act_l_midy = temp_l_midpoint
      
      nor_l_x, nor_l_y = abs(img_midpt_x-act_l_midx), abs(img_midpt_y-act_l_midy)

      if act_l_midx>img_midpt_x:
        nor_l_x = -nor_l_x
      if act_l_midy>img_midpt_y:
        nor_l_y = -nor_l_y

      if len(LOCKED_position)>10:
        cv2.line(angle_image, (LOCKED_position[4][0]+nor_l_x, LOCKED_position[4][1]+nor_l_y), (LOCKED_position[8][0]+nor_l_x, LOCKED_position[8][1]+nor_l_y), (50,50,255), 2)
        cv2.circle(angle_image, (LOCKED_position[4][0]+nor_l_x, LOCKED_position[4][1]+nor_l_y), 5, (0,0,255), 2)
        cv2.circle(angle_image, (LOCKED_position[8][0]+nor_l_x, LOCKED_position[8][1]+nor_l_y), 5, (0,0,255), 2)
        
        l_anglex, l_angley = LOCKED_position[8][0]+nor_l_x, LOCKED_position[8][1]+nor_l_y
    except:
      pass


########################################################################## angle_image section2
    angle_image = cv2.flip(angle_image, 1)

    try:
      cv2.putText(angle_image, "fps:"+str(fps), (50, 50), cv2.FONT_ITALIC, 0.8, (0,0,255), 2)
    except:
      pass

    try:
      t_text = str(t_distance)
      textsize = cv2.getTextSize(t_text, cv2.FONT_ITALIC, 1, 2)[0]
      textX = (angle_image.shape[1] - textsize[0]) // 2
      textY = (angle_image.shape[0] + textsize[1]) // 2

      t_midx, t_midy = temp_midpoint
      cv2.putText(angle_image, str(t_distance), (textX+10, adi_height//2+150), cv2.FONT_ITALIC, 0.8, (0,0,255), 2)

    except:
      cv2.putText(angle_image, "-Undetected-", (adi_width//2-85, adi_height//2+150), cv2.FONT_ITALIC, 0.8, (255,0,0), 2)


    try:
      #radian
      t_angle = cal_three_point_angle(anglex, img_midpt_x-1, angley, 1, img_midpt_x, img_midpt_y)
      t_angle_degree = (t_angle/math.pi*180)
      t_angle_degree = round(t_angle_degree, 6)
      if t_angle_degree != 52.861824:
        t_angle_degree = round(t_angle_degree, 3)
        cv2.putText(angle_image, str(t_angle_degree), (adi_width//2-55, adi_height//2+180), cv2.FONT_ITALIC, 0.8, (0,255,0), 2)
      else:
        cv2.putText(angle_image, "-Undetected-", (adi_width//2-85, adi_height//2+180), cv2.FONT_ITALIC, 0.8, (255,0,0), 2)
      
    except:
      pass

    t_l_angle_degree = -1
    try:
      #radian
      t_l_angle = cal_three_point_angle(anglex, l_anglex, angley, l_angley, img_midpt_x, img_midpt_y)
      t_l_angle_degree = (t_l_angle/math.pi*180)
      if t_l_angle_degree < 0:
        t_l_angle_degree += 180
      t_l_angle_degree = round(t_l_angle_degree, 3)
      if t_l_angle_degree != 0.0:
        cv2.putText(angle_image, str(t_l_angle_degree), (adi_width//2-55, adi_height//2+210), cv2.FONT_ITALIC, 0.8, (0,255,0), 2)
      else:
        cv2.putText(angle_image, "-Undetected-", (adi_width//2-85, adi_height//2+210), cv2.FONT_ITALIC, 0.8, (255,0,0), 2)
      
    except:
      pass


    #### 임시
    t_lock_distance = -1
    try:
      #print(LOCKED_position[8], landmarks_coordinates[8])
      if len(LOCKED_position) > 10:
        t_lock_distance = cal_distance(LOCKED_position[4][0], LOCKED_position[8][0], LOCKED_position[4][1], LOCKED_position[8][1])
        t_lock_distance = round(t_lock_distance, 3)
    except:
      pass

    try:
      #print(t_l_distance, t_distance)
      if t_lock_distance > 0:
        check_zoom_inout(t_lock_distance, t_distance)
      if len(landmarks_coordinates) < 10:
        key_turn_off('pageup')
        key_turn_off('pagedown')
    except:
      pass
    
    try:
      if len(landmarks_coordinates)>10 and len(LOCKED_position)>10:
        check_rotate_move(l_anglex, anglex, t_l_angle_degree)
    except:
      pass
    
##########################################################################

    # Lock and Unlock previous position
    #print(LOCKED_position)
    if FIND_CHECK:
      lock_end = False
      if not lock_check:
        if lock_start:
          lock_time_check = time.time()
          lock_time = lock_time_check - lock_time_start
          if lock_time > LOCK_LIMIT_TIME:
            LOCKED_position = landmarks_coordinates.copy()
            lock_check = True
        else:
          lock_time_start = time.time()
          lock_start = True
    else:
      lock_start = False
      if lock_check:
        if lock_end:
          lock_time_check = time.time()
          lock_time = lock_time_check - lock_time_start
          if lock_time > LOCK_LIMIT_TIME:
            LOCKED_position = []
            lock_check = False
        else:
          lock_time_start = time.time()
          lock_end = True



##########################################################################
    
    cv2.imshow('sub', gray_image)
    cv2.imshow('debug', angle_image)

    key_input = cv2.waitKey(1)
    
    
    # 27:ESC, 32:SPACEBAR
    if key_input & 0xFF == 27:
      break
    elif key_input & 0xFF == 32:
      SYSMODE = (SYSMODE+1) % 3
      print(SYSMODE)
    else:
      if key_input & 0xFF == 255 or key_input & 0xFF == 0:
        pass
      else:
        #print(key_input & 0xFF)
        pass

    FIND_CHECK = False

cap.release()