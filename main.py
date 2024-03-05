import time
# import pyttsx3
import cv2
import mediapipe as mp
import numpy as np
import os

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# engine = pyttsx3.init()

def calculate_calories_burned(mets, weight_kg, duration_minutes):
    calories_per_minute = (mets * 3.5 * weight_kg) / 200
    total_calories_burned = calories_per_minute * duration_minutes
    return total_calories_burned


def calculate_angle(a, b, c):
    import numpy as np

    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def speak(text):
    # engine.say(text)
    # engine.runAndWait()
    os.system(f"say '{text}'")


def track_biceps_curls(side, weight, MET):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Variables to keep track of counts
    curl_start_time = None
    total_curl_duration = 0
    biceps_curls_count = 0
    stage = "down"

    ret, frame = cap.read()
    if ret:
        frame_width = frame.shape[1]
    else:
        frame_width = 640  # Default width in case of failure to capture a frame

    # Define the right margin for the progress bar
    right_margin = 50
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and find poses
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get landmarks for biceps curls and squats tracking
            landmarks = results.pose_landmarks.landmark
            angle = 0
            if side == "2":
                # Example for biceps curls: calculate angle between shoulder, elbow, and wrist
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate the angle
                angle = calculate_angle(shoulder, elbow, wrist)

            elif side == "1":
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate the angle
                angle = calculate_angle(shoulder, elbow, wrist)

            # Progress bar logic
            bar = np.interp(angle, (30, 160), (650, 100))
            progress_bar_left = frame_width - right_margin - 75  # Calculate left x-coordinate for the progress bar
            cv2.rectangle(frame, (progress_bar_left, 100), (progress_bar_left + 75, 650), (0, 255, 0), 3)
            cv2.rectangle(frame, (progress_bar_left, int(bar)), (progress_bar_left + 75, 650), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f'{int(np.interp(angle, (30, 160), (0, 100)))}%', (progress_bar_left, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if angle > 160 and stage == "up":
                stage = "down"
                state_text = "Move down"  # Guide user to move down from the 'up' position
                if curl_start_time is None:  # Start time for a new curl
                    curl_start_time = time.time()
            elif angle < 30 and stage == "down":
                if curl_start_time is not None:
                    curl_end_time = time.time()
                    curl_duration = curl_end_time - curl_start_time
                    total_curl_duration += curl_duration  # Accumulate total duration
                    curl_start_time = None  # Reset for the next curl
                stage = "up"
                state_text = "Great!"  # Positive reinforcement for completing the curl
                biceps_curls_count += 1
            else:
                if stage == "down":
                    state_text = "Move up"  # Guide user to return to the 'up' position to complete the curl
                else:
                    state_text = "Curl more"  # Encourage the user to continue the curl motion if not at the extremes

            # Display the state text above the progress bar
            cv2.putText(frame, state_text, (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
            # Display the count
            cv2.putText(frame, 'Biceps Curls: ' + str(biceps_curls_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), 4, cv2.LINE_AA)
        # Show the frame
        cv2.imshow('AI Workout Tracker', frame)
        # Break the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            total_curl_duration = total_curl_duration / (60)
            print("Workout session ended. Duration(mins): ", total_curl_duration)
            calories_burned = calculate_calories_burned(MET, weight, total_curl_duration)
            print(calories_burned)

            pose.close()
            cap.release()
            cv2.destroyAllWindows()
            return biceps_curls_count, total_curl_duration, calories_burned


def track_squats(weight, MET):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Variables to keep track of squats
    squats_count = 0
    stage = "up"  # Start assuming the person is standing
    squat_start_time = None
    total_squat_duration = 0

    # Get the width of the frame from the video capture object
    ret, frame = cap.read()
    if ret:
        frame_width = frame.shape[1]
    else:
        frame_width = 640  # Default width in case of failure to capture a frame

    # Define the right margin for the progress bar
    right_margin = 50

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            # Get landmarks for squats tracking
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate the angle
            angle = calculate_angle(hip, knee, ankle)

            # Progress bar logic
            bar_height = np.interp(angle, (60, 160), (100, 650))  # Adjust these values as needed

            # Calculate left x-coordinate for the progress bar
            progress_bar_left = frame_width - right_margin - 75

            # Draw the progress bar background
            cv2.rectangle(frame, (progress_bar_left, 100), (progress_bar_left + 75, 650), (0, 255, 0), 3)

            # Fill the progress bar based on the squat depth
            cv2.rectangle(frame, (progress_bar_left, int(bar_height)), (progress_bar_left + 75, 650), (0, 255, 0),
                          cv2.FILLED)

            # Display the percentage of squat depth
            cv2.putText(frame, f'{int(np.interp(angle, (60, 160), (100, 0)))}%', (progress_bar_left, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


            if angle > 160:
                state_text = "Squat down"
                if stage == "up":  # Squat starts when moving from "up" to "down"
                    squat_start_time = time.time()  # Start timing the squat
                stage = "down"
            elif angle < 90:
                state_text = "Stand up"
                if stage == "down" and squat_start_time is not None:  # Squat ends when moving from "down" to "up"
                    squat_end_time = time.time()
                    squat_duration = squat_end_time - squat_start_time
                    total_squat_duration += squat_duration  # Accumulate total squat duration
                    squat_start_time = None  # Reset for the next squat
                stage = "up"
                squats_count += 1
            else:
                state_text = "Hold"

            cv2.putText(frame, state_text, (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

            # Display the count
            cv2.putText(frame, 'Squats: ' + str(squats_count), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)

        cv2.imshow('AI Workout Tracker - Squats', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            total_squat_duration = total_squat_duration / (60)
            print("Workout session ended. Duration(mins): ", total_squat_duration)
            calories_burned = calculate_calories_burned(MET, weight, total_squat_duration)
            print(calories_burned)

            pose.close()
            cap.release()
            cv2.destroyAllWindows()
            return squats_count, total_squat_duration, calories_burned


def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    distance = np.linalg.norm(a - b)
    return distance


def track_lunges(weight, MET):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Variables to keep track of side lunges
    side_lunges_count = 0
    stage = "up"  # Start assuming the person is standing
    total_lunge_duration = 0
    lunge_start_time = None

    ret, frame = cap.read()
    if ret:
        frame_width = frame.shape[1]
    else:
        frame_width = 640  # Default width in case of failure to capture a frame

    # Define the right margin for the progress bar
    right_margin = 50

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            # Get landmarks for side lunges tracking
            hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate the angle for the left and right legs
            angle_left = calculate_angle(hip_left, knee_left, ankle_left)
            angle_right = calculate_angle(hip_right, knee_right, ankle_right)

            # Choose the angle based on the leg with a larger bend, indicating the lunging leg
            angle = angle_left if angle_left < angle_right else angle_right

            # Calculate the distance between the feet
            distance_between_feet = calculate_distance(ankle_left, ankle_right)

            if distance_between_feet > 0.25:  # Assuming a wider stance indicates the start of a side lunge
                feedback_text = "Bend your knees"  # Starting the side lunge
                if lunge_start_time is None:  # Start time for a new lunge
                    lunge_start_time = time.time()
            elif distance_between_feet < 0.15:  # Assuming a narrower stance indicates the end of a side lunge
                feedback_text = "Stand up"  # Completing the side lunge
                if lunge_start_time is not None:
                    lunge_end_time = time.time()
                    lunge_duration = lunge_end_time - lunge_start_time
                    total_lunge_duration += lunge_duration  # Accumulate total duration
                    lunge_start_time = None  # Reset for the next lunge
            else:
                feedback_text = "Good"

            # Display the feedback text
            cv2.putText(frame, feedback_text, (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                        cv2.LINE_AA)

            # Adjust the distance thresholds for leniency
            bar_height = np.interp(distance_between_feet, [0.05, 0.4], [650, 100])  # Reduced lower bound

            # Progress bar logic remains the same
            progress_bar_left = frame_width - right_margin - 75
            cv2.rectangle(frame, (progress_bar_left, 100), (progress_bar_left + 75, 650), (0, 255, 0), 3)
            cv2.rectangle(frame, (progress_bar_left, int(bar_height)), (progress_bar_left + 75, 650), (0, 255, 0),
                          cv2.FILLED)

            # Display the progress as a percentage based on the adjusted distance between feet
            cv2.putText(frame, f'{int(np.interp(distance_between_feet, [0.05, 0.4], [0, 100]))}%',
                        (progress_bar_left, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the count
            cv2.putText(frame, 'Side Lunges: ' + str(side_lunges_count), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('AI Workout Tracker - Side Lunges', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            total_lunge_duration = total_lunge_duration / (60)
            print("Workout session ended. Duration(mins): ", total_lunge_duration)
            calories_burned = calculate_calories_burned(MET, weight, total_lunge_duration)
            print(calories_burned)

            pose.close()
            cap.release()
            cv2.destroyAllWindows()
            return side_lunges_count, total_lunge_duration, calories_burned


def generate_report(data):
    report = "\nWorkout Report\n\n"
    for exercise, info in data.items():
        report += f"{exercise}:\n"
        report += f" - Total Repetitions: {info['Count']}\n"
        report += f" - Total Duration: {info['Duration'] * 60:.2f} minutes\n"  # Convert hours to minutes
        report += f" - Estimated Calories Burned: {info['Calories Burned']:.2f} calories\n\n"
    return report


def send_email(report, recipient_email):
    sender_email = "kappi.kothi@gmail.com"
    sender_password = "Kruthi@98"

    # Set up the email server and login
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "Your Workout Report"
    msg.attach(MIMEText(report, 'plain'))

    # Send the email and close the server
    server.send_message(msg)
    server.quit()


def main():
    report_data = {}
    weight = int(input("Weight(kg):"))
    while True:
        choice = int(input("Choose the exercise: 1. Biceps Curls  2. Squats   3. Side Lunges    4.Exit"))
        if choice == 1:
            side = input("Choose the side: 1. Left  2. Right ")
            count, duration, calories = track_biceps_curls(side, weight, 3.0)
            report_data['Biceps Curls'] = {'Count': count, 'Duration': duration, 'Calories Burned': calories}
        elif choice == 2:
            count, duration, calories = track_squats(weight, 3.0)
            report_data['Squats'] = {'Count': count, 'Duration': duration, 'Calories Burned': calories}
        elif choice == 3:
            count, duration, calories = track_lunges(weight, 3.0)
            report_data['Side Lunges'] = {'Count': count, 'Duration': duration, 'Calories Burned': calories}
        elif choice == 4:
            print("Exiting workout session.")
            break
        else:
            print("Invalid choice. Please try again.")

    report = generate_report(report_data)
    print(report)
    # Ask for the user's email address
    ch = input("Do you want the report to be sent over email? (Y/N)")
    if ch.upper() == 'Y':
        recipient_email = input("Enter your email address to receive the report: ")

        # Send the report
        send_email(report, recipient_email)

        print("Report sent to your email.")
    else:
        print("Session ended")


if __name__ == "__main__":
    main()
