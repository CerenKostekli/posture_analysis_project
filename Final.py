import csv
import cv2
import numpy as np
import time
import math as m
import mediapipe as mp
import pygame



#Sound Effect
def sesli_uyari():
    pygame.init()
    pygame.mixer.init()
    ses_dosyasi = r"C:\Users\ceren\OneDrive\Masaüstü\Workspace\Pose_Estimation\posture_analysis\start-13691.mp3"
    uyari_sesi = pygame.mixer.Sound(ses_dosyasi)
    return uyari_sesi

uyari_sesi = sesli_uyari()
# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

data = ['class','neck', 'torso', 'time']

with open('coords1.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting= csv.QUOTE_MINIMAL)
    csv_writer.writerow(data)

def export_landmark(results, action):
    try:
        keypoints = np.array([results]).flatten().tolist()
        keypoints.insert(0, action)
        with open ('coords1.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting= csv.QUOTE_MINIMAL)
            csv_writer.writerow(keypoints)
    except Exception as e:
        pass

"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""

def sendWarning(x):
    pass

# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)
warn = "Lutfen kameraya yan acida durunuz."
(text_width, text_height), baseline = cv2.getTextSize(warn, font, 0.9, 2)
# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ===============================================================================================#


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Process the image.
        results = pose.process(image)

        # Convert the image back to BGR.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Use lm and lmPose as representative of the following methods.
        lmPose = mp_pose.PoseLandmark

        # Acquire the landmark coordinates.
        # Once aligned properly, left or right should not be a concern.
        try: 
            lm = results.pose_landmarks.landmark 

            l_shldr_x = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w)
            l_shldr_y = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)   
            r_shldr_x = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w)
            r_shldr_y = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)
        
            #Left side
            l_ear_x = int(lm[mp_pose.PoseLandmark.LEFT_EAR.value].x * w)
            l_ear_y = int(lm[mp_pose.PoseLandmark.LEFT_EAR.value].y * h)
            l_hip_x = int(lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w)
            l_hip_y = int(lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)

            #Right Side
            r_ear_x = int(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w)
            r_ear_y = int(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h)
            r_hip_x = int(lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w)
            r_hip_y = int(lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)

            #to determine which side 
            right_shoulder  = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
            left_shoulder  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z

            # Calculate distance between left shoulder and right shoulder points.
            offset_L = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            offset_R = findDistance(r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y)

            # Calculate angles.
            neck_inclination_l = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination_l = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            neck_inclination_r = findAngle(r_shldr_x, r_shldr_y, r_ear_x, r_ear_y)
            torso_inclination_r = findAngle(r_hip_x, r_hip_y, r_shldr_x, r_shldr_y)

            # Assist to align the camera to point at the side view of the person.
            # Offset threshold 30 is based on results obtained from analysis over 100 samples.
            if offset_L < 100 or offset_R<100:
                cv2.putText(image, str(int(offset_L)) + ' Hizali', (w - 150, 30), font, 0.9, green, 2)
                cv2.putText(image, str(int(offset_R)) + ' Hizali', (w - 150, 30), font, 0.9, green, 2)
                if left_shoulder < right_shoulder:

                    # Draw landmarks.
                    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
                    cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

                    # Let's take y - coordinate of P3 100px above x1,  for display elegance.
                    # Although we are taking y = 0 while calculating angle between P1,P2,P3.
                    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
                    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
                    cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

                    # Similarly, here we are taking y - coordinate 100px above x1. Note that
                    # you can take any value for y, not necessarily 100 or 200 pixels.
                    cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

                    # Put text, Posture and angle inclination.
                    # Text string for display.
                    angle_text_string = 'Boyun : ' + str(int(neck_inclination_l)) + '  Govde : ' + str(int(torso_inclination_l))

                    # Determine whether good posture or bad posture.
                    # The threshold angles have been set based on intuition.
                    if neck_inclination_l < 40 and torso_inclination_l < 10:
                        bad_frames = 0
                        good_frames += 1

                        # Calculate the time of remaining in a particular posture.
                        good_time = (1 / fps) * good_frames
                        bad_time =  (1 / fps) * bad_frames

                        angle=  [int(neck_inclination_l), int(torso_inclination_l), good_time]
                        export_landmark(angle,'good frame')
                        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
                        cv2.putText(image, str(int(neck_inclination_l)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
                        cv2.putText(image, str(int(torso_inclination_l)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

                        # Join landmarks.
                        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
                        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
                        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
                        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

                    else:
                        good_frames = 0
                        bad_frames += 1

                        # Calculate the time of remaining in a particular posture.
                        good_time = (1 / fps) * good_frames
                        bad_time =  (1 / fps) * bad_frames

                        angle=  [int(neck_inclination_l), int(torso_inclination_l), bad_time]
                        export_landmark(angle,'bad frame')
                        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
                        cv2.putText(image, str(int(neck_inclination_l)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
                        cv2.putText(image, str(int(torso_inclination_l)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

                        # Join landmarks.
                        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
                        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
                        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
                        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)


                    # Pose time.
                    if good_time > 0:
                        time_string_good = 'Dogru Durus Zamani : ' + str(round(good_time, 1)) + 's'
                        cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
                    else:
                        time_string_bad = 'Yanlis Durus Zamani : ' + str(round(bad_time, 1)) + 's'
                        cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)
                        uyari_sesi.play();
                    # If you stay in bad posture for more than 3 minutes (180s) send an alert.
                    if bad_time > 180:
                        sendWarning()
                    # Write frames.
                    #video_output.write(image)
                
                else:
                
                    
                
                    # Draw landmarks.
                    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, yellow, -1)
                    cv2.circle(image, (r_ear_x, r_ear_y), 7, yellow, -1)

                    # Let's take y - coordinate of P3 100px above x1,  for display elegance.
                    # Although we are taking y = 0 while calculating angle between P1,P2,P3.
                    cv2.circle(image, (r_shldr_x, r_shldr_y - 100), 7, yellow, -1)
                    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, pink, -1)
                    cv2.circle(image, (r_hip_x, r_hip_y), 7, yellow, -1)

                    # Similarly, here we are taking y - coordinate 100px above x1. Note that
                   # you can take any value for y, not necessarily 100 or 200 pixels.
                    cv2.circle(image, (r_hip_x, r_hip_y - 100), 7, yellow, -1)

                    # Put text, Posture and angle inclination.
                    # Text string for display.
                    angle_text_string = 'Boyun : ' + str(int(neck_inclination_r)) + '  Govde : ' + str(int(torso_inclination_r))

                    if neck_inclination_r < 40 and torso_inclination_r < 10:
                        bad_frames = 0
                        good_frames += 1

                        # Calculate the time of remaining in a particular posture.
                        good_time = (1 / fps) * good_frames
                        bad_time =  (1 / fps) * bad_frames

                        angle=  [int(neck_inclination_r), int(torso_inclination_r), good_time]
                        export_landmark(angle,'good')
                        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
                        cv2.putText(image, str(int(neck_inclination_r)), (r_shldr_x + 10, r_shldr_y), font, 0.9, light_green, 2)
                        cv2.putText(image, str(int(torso_inclination_r)), (r_hip_x + 10, r_hip_y), font, 0.9, light_green, 2)

                        # Join landmarks.
                        cv2.line(image, (r_shldr_x, r_shldr_y), (r_ear_x, r_ear_y), green, 4)
                        cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y - 100), green, 4)
                        cv2.line(image, (r_hip_x, r_hip_y), (r_shldr_x, r_shldr_y), green, 4)
                        cv2.line(image, (r_hip_x, r_hip_y), (r_hip_x, r_hip_y - 100), green, 4)

                    else:
                        good_frames = 0
                        bad_frames += 1

                        # Calculate the time of remaining in a particular posture.
                        good_time = (1 / fps) * good_frames
                        bad_time =  (1 / fps) * bad_frames

                        angle=  [int(neck_inclination_r), int(torso_inclination_r), bad_time]
                        export_landmark(angle,'bad')

                        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
                        cv2.putText(image, str(int(neck_inclination_r)), (r_shldr_x + 10, r_shldr_y), font, 0.9, red, 2)
                        cv2.putText(image, str(int(torso_inclination_r)), (r_hip_x + 10, r_hip_y), font, 0.9, red, 2)

                        # Join landmarks.
                        cv2.line(image, (r_shldr_x, r_shldr_y), (r_ear_x, r_ear_y), red, 4)
                        cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y - 100), red, 4)
                        cv2.line(image, (r_hip_x, r_hip_y), (r_shldr_x, r_shldr_y), red, 4)
                        cv2.line(image, (r_hip_x, r_hip_y), (r_hip_x, r_hip_y - 100), red, 4)     

                    # Pose time.
                    if good_time > 0:
                        time_string_good = 'Dogru Durus Zamani : ' + str(round(good_time, 1)) + 's'
                        cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
                    else:
                        time_string_bad = 'Yanlis Durus Zamani : ' + str(round(bad_time, 1)) + 's'
                        cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)
                        uyari_sesi.play();

                    # If you stay in bad posture for more than 3 minutes (180s) send an alert.
                    if bad_time > 180:
                        sendWarning()
                    # Write frames.
                    #video_output.write(image)
                    
            else:
                cv2.putText(image, warn, (image.shape[1]- text_width, 30), font, 0.9, red, 2)
                uyari_sesi.play()      
        
        except:
            pass
        # Display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
