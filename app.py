import cv2
import mediapipe as mp
import numpy as np
import tempfile
import streamlit as st
import time


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_calories_burned(mets, weight_kg, duration_minutes):
    calories_per_minute = (mets * 3.5 * weight_kg) / 200
    total_calories_burned = calories_per_minute * duration_minutes
    return total_calories_burned

def start_webcam():
    st.session_state.use_webcam = True

def stop_webcam():
    st.session_state.use_webcam = False
    st.session_state.camera_active = False  # Signal to stop the camera



def track_biceps_curls(side, weight, MET):
    # Initialize or use existing session state variables
    if 'biceps_curls_count' not in st.session_state:
        st.session_state.biceps_curls_count = 0
    if 'total_curl_duration' not in st.session_state:
        st.session_state.total_curl_duration = 0
    if 'calories_burned' not in st.session_state:
        st.session_state.calories_burned = 0
    if 'stage' not in st.session_state:
        st.session_state.stage = "down"
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'curl_start_time' not in st.session_state:
        st.session_state['curl_start_time'] = None
    if 'curl_end_time' not in st.session_state:
        st.session_state['curl_end_time'] = None

    right_margin = 50
    stframe = st.empty()
    st.button('Use Webcam', on_click=start_webcam)

    curl_start_time = None

    if st.session_state.get('use_webcam', False):
        st.session_state.camera_active = True
        st.button('Stop Webcam', on_click=stop_webcam)

        vid = cv2.VideoCapture(0)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while vid.isOpened() and st.session_state.camera_active:
                ret, image = vid.read()
                if not ret:
                    break

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if results.pose_landmarks:
                    # Draw pose landmarks on the frame
                    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    # Get landmarks for biceps curls and squats tracking
                    landmarks = results.pose_landmarks.landmark
                    angle = 0
                    visible_threshold = 0.5
                    st.session_state.ready_for_rep = False
                    if side == "Right":
                        # Example for biceps curls: calculate angle between shoulder, elbow, and wrist
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        shoulder_visible = landmarks[
                                               mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > visible_threshold
                        elbow_visible = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > visible_threshold
                        wrist_visible = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > visible_threshold

                        if shoulder_visible and elbow_visible and wrist_visible:
                            angle = calculate_angle(shoulder, elbow, wrist)
                            st.session_state.ready_for_rep = True
                    elif side == "Left":
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        shoulder_visible = landmarks[
                                               mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > visible_threshold
                        elbow_visible = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > visible_threshold
                        wrist_visible = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > visible_threshold
                        # Calculate the angle
                        if shoulder_visible and elbow_visible and wrist_visible:
                            angle = calculate_angle(shoulder, elbow, wrist)
                            st.session_state.ready_for_rep = True

                    if st.session_state.ready_for_rep:
                        # Progress bar logic
                        bar = np.interp(angle, (30, 160), (650, 100))
                        progress_bar_left = width - right_margin - 75  # Calculate left x-coordinate for the progress bar
                        cv2.rectangle(image, (progress_bar_left, 100), (progress_bar_left + 75, 650), (0, 255, 0), 3)
                        cv2.rectangle(image, (progress_bar_left, int(bar)), (progress_bar_left + 75, 650), (0, 255, 0),
                                      cv2.FILLED)
                        cv2.putText(image, f'{int(np.interp(angle, (30, 160), (0, 100)))}%', (progress_bar_left, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        if angle > 160 and st.session_state.stage == "up":
                            st.session_state.stage = "down"
                            state_text = "Move down"  # Guide user to move down from the 'up' position
                            if st.session_state['curl_start_time'] is None:  # Curl started
                                st.session_state['curl_start_time'] = time.time()
                        elif angle < 30 and st.session_state.stage == "down":
                            if st.session_state['curl_start_time'] is not None:
                                st.session_state['curl_end_time'] = time.time()
                                curl_duration = st.session_state['curl_end_time'] - st.session_state['curl_start_time']
                                st.session_state['total_curl_duration'] += curl_duration
                                st.session_state['curl_start_time'] = None
                            st.session_state.stage = "up"
                            state_text = "Great!"  # Positive reinforcement for completing the curl
                            st.session_state.biceps_curls_count += 1
                        else:
                            if st.session_state.stage == "down":
                                state_text = "Move up"  # Guide user to return to the 'up' position to complete the curl
                            else:
                                state_text = "Move down"
                                # state_text = "Curl more"  # Encourage the user to continue the curl motion if not at the extremes

                        # Display the state text above the progress bar
                        cv2.putText(image, state_text, (50, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                    2,
                                    cv2.LINE_AA)
                        # Display the count
                        cv2.putText(image, 'Biceps Curls: ' + str(st.session_state.biceps_curls_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    2,
                                    (255, 255, 255), 4, cv2.LINE_AA)

                stframe.image(image, channels="BGR", use_column_width=True)

                if not st.session_state.get('use_webcam', False):
                    # When the stop button is pressed
                    # st.session_state['total_curl_duration'] = st.session_state['total_curl_duration']
                    print(st.session_state['total_curl_duration'])
                    st.session_state.calories_burned = calculate_calories_burned(MET, weight,
                                                                                 st.session_state['total_curl_duration'] / 60)  # Convert to minutes for calculation
                    break

            vid.release()
            cv2.destroyAllWindows()
            st.success('Video is Processed')

    # Assuming tracking logic populates these variables correctly
    return st.session_state.biceps_curls_count, st.session_state.total_curl_duration, st.session_state.calories_burned


def generate_report(data):
    report = "\nWorkout Report\n\n"
    for exercise, info in data.items():
        report += f"{exercise}:\n"
        report += f" - Total Repetitions: {info['Count']}\n"
        report += f" - Total Duration: {info['Duration']:.2f} seconds\n"
        report += f" - Estimated Calories Burned: {info['Calories Burned']:.2f} calories\n\n"
    return report


def track_squats(weight, MET):
    # Initialize or use existing session state variables
    if 'squats_count' not in st.session_state:
        st.session_state.squats_count = 0
    if 'total_squat_duration' not in st.session_state:
        st.session_state.total_squat_duration = 0
    if 'calories_burned' not in st.session_state:
        st.session_state.calories_burned = 0
    if 'stage' not in st.session_state:
        st.session_state.stage = "up"
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'squat_start_time' not in st.session_state:
        st.session_state['squat_start_time'] = None
    if 'squat_end_time' not in st.session_state:
        st.session_state['squat_end_time'] = None

    right_margin = 50
    stframe = st.empty()
    st.button('Use Webcam', on_click=start_webcam)

    if st.session_state.get('use_webcam', False):
        st.session_state.camera_active = True
        st.button('Stop Webcam', on_click=stop_webcam)

        vid = cv2.VideoCapture(0)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while vid.isOpened() and st.session_state.camera_active:
                ret, image = vid.read()
                if not ret:
                    break

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if results.pose_landmarks:
                    # Draw pose landmarks on the frame
                    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    # Get landmarks for biceps curls and squats tracking
                    landmarks = results.pose_landmarks.landmark
                    visible_threshold = 0.5
                    st.session_state.ready_for_rep = False

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    hip_visible = landmarks[
                                      mp_pose.PoseLandmark.LEFT_HIP.value].visibility > visible_threshold
                    knee_visible = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > visible_threshold
                    ankle_visible = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > visible_threshold

                    if hip_visible and knee_visible and ankle_visible:
                        angle = calculate_angle(hip, knee, ankle)
                        st.session_state.ready_for_rep = True

                    # Progress bar logic
                    if st.session_state.ready_for_rep:
                        bar_height = np.interp(angle, (60, 160), (100, 650))  # Adjust these values as needed

                        # Calculate left x-coordinate for the progress bar
                        progress_bar_left = width - right_margin - 75

                        # Draw the progress bar background
                        cv2.rectangle(image, (progress_bar_left, 100), (progress_bar_left + 75, 650), (0, 255, 0), 3)

                        # Fill the progress bar based on the squat depth
                        cv2.rectangle(image, (progress_bar_left, int(bar_height)), (progress_bar_left + 75, 650),
                                      (0, 255, 0),
                                      cv2.FILLED)

                        # Display the percentage of squat depth
                        cv2.putText(image, f'{int(np.interp(angle, (60, 160), (100, 0)))}%', (progress_bar_left, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        if angle > 160:
                            state_text = "Squat down"
                            if st.session_state.stage == "up":  # Squat starts when moving from "up" to "down"
                                st.session_state["squat_start_time"] = time.time()  # Start timing the squat
                            st.session_state.stage = "down"
                        elif angle < 90:
                            state_text = "Stand up"
                            if st.session_state.stage == "down" and st.session_state["squat_start_time"] is not None:  # Squat ends when moving from "down" to "up"
                                st.session_state["squat_end_time"] = time.time()
                                squat_duration = st.session_state["squat_end_time"] - st.session_state["squat_start_time"]
                                st.session_state["total_squat_duration"] += squat_duration  # Accumulate total squat duration
                                st.session_state["squat_start_time"] = None  # Reset for the next squat
                            st.session_state.stage = "up"
                            st.session_state.squats_count += 1
                        else:
                            state_text = "Hold"

                        cv2.putText(image, state_text, (50, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2,
                                    cv2.LINE_AA)

                        # Display the count
                        cv2.putText(image, 'Squats: ' + str(st.session_state.squats_count), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255),
                                    2, cv2.LINE_AA)
                stframe.image(image, channels="BGR", use_column_width=True)

                if not st.session_state.get('use_webcam', False):
                    # When the stop button is pressed
                    # st.session_state['total_curl_duration'] = st.session_state['total_curl_duration']
                    st.session_state.calories_burned = calculate_calories_burned(MET, weight,
                                                                                 st.session_state[
                                                                                     'total_curl_duration'] / 60)  # Convert to minutes for calculation
                    break

            vid.release()
            cv2.destroyAllWindows()
            st.success('Video is Processed')

            # Assuming tracking logic populates these variables correctly
    return st.session_state.squats_count, st.session_state.total_squat_duration, st.session_state.calories_burned

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    distance = np.linalg.norm(a - b)
    return distance

def track_lunges(weight, MET):
    if 'side_lunges_count' not in st.session_state:
        st.session_state.side_lunges_count = 0
    if 'total_lunge_duration' not in st.session_state:
        st.session_state.total_lunge_duration = 0
    if 'calories_burned' not in st.session_state:
        st.session_state.calories_burned = 0
    if 'stage' not in st.session_state:
        st.session_state.stage = "down"
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'lunge_start_time' not in st.session_state:
        st.session_state['lunge_start_time'] = None
    if 'lunge_end_time' not in st.session_state:
        st.session_state['lunge_end_time'] = None

    right_margin = 50
    stframe = st.empty()
    st.button('Use Webcam', on_click=start_webcam)

    if st.session_state.get('use_webcam', False):
        st.session_state.camera_active = True
        st.button('Stop Webcam', on_click=stop_webcam)

        vid = cv2.VideoCapture(0)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while vid.isOpened() and st.session_state.camera_active:
                ret, image = vid.read()
                if not ret:
                    break

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if results.pose_landmarks:
                    # Draw pose landmarks on the frame
                    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    # Get landmarks for biceps curls and squats tracking
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
                        if st.session_state["lunge_start_time"] is None:  # Start time for a new lunge
                            st.session_state["lunge_start_time"] = time.time()
                    elif distance_between_feet < 0.15:  # Assuming a narrower stance indicates the end of a side lunge
                        feedback_text = "Stand up"  # Completing the side lunge
                        if st.session_state["lunge_start_time"] is not None:
                            st.session_state["lunge_end_time"] = time.time()
                            lunge_duration = st.session_state["lunge_end_time"] - st.session_state["lunge_start_time"]
                            st.session_state["total_lunge_duration"] += lunge_duration  # Accumulate total duration
                            st.session_state["lunge_start_time"] = None  # Reset for the next lunge
                            st.session_state.side_lunges_count += 1
                    else:
                        feedback_text = "Good"

                    # Display the feedback text
                    cv2.putText(image, feedback_text, (50, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2,
                                cv2.LINE_AA)

                    # Adjust the distance thresholds for leniency
                    bar_height = np.interp(distance_between_feet, [0.05, 0.4], [650, 100])  # Reduced lower bound

                    # Progress bar logic remains the same
                    progress_bar_left = width - right_margin - 75
                    cv2.rectangle(image, (progress_bar_left, 100), (progress_bar_left + 75, 650), (0, 255, 0), 3)
                    cv2.rectangle(image, (progress_bar_left, int(bar_height)), (progress_bar_left + 75, 650),
                                  (0, 255, 0),
                                  cv2.FILLED)

                    # Display the progress as a percentage based on the adjusted distance between feet
                    cv2.putText(image, f'{int(np.interp(distance_between_feet, [0.05, 0.4], [0, 100]))}%',
                                (progress_bar_left, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Display the count
                    cv2.putText(image, 'Side Lunges: ' + str(st.session_state.side_lunges_count), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)
                stframe.image(image, channels="BGR", use_column_width=True)

                if not st.session_state.get('use_webcam', False):
                    # When the stop button is pressed
                    # st.session_state['total_curl_duration'] = st.session_state['total_curl_duration']
                    st.session_state['calories_burned'] = calculate_calories_burned(MET, weight,
                                                                                    st.session_state[
                                                                                        'total_curl_duration'] / 60)  # Convert to minutes for calculation
                    break

            vid.release()
            cv2.destroyAllWindows()
            st.success('Video is Processed')

            # Assuming tracking logic populates these variables correctly
    return st.session_state.side_lunges_count, st.session_state.total_lunge_duration, st.session_state.calories_burned

from PIL import Image

def load_and_resize_image(image_path, width=None):
    image = Image.open(image_path)
    if width is not None:
        # Calculate new height to maintain aspect ratio
        aspect_ratio = image.height / image.width
        new_height = int(aspect_ratio * width)
        image = image.resize((width, new_height))
    return image
def main():
    st.title("Workout Tracker with Real-time Pose Estimation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('1025.gif', caption='Biscep Curls')
    with col2:
        st.image('Jump_Squats.gif', caption='Squats')
    with col3:
        st.image('side_lunges.gif', caption='Side Lunges')

    weight = st.number_input("Enter your weight (kg):", min_value=1, max_value=200, value=70, step=1)

    exercise_options = ["None","Biceps Curls", "Squats", "Side Lunges"]
    exercise_selection = st.selectbox("Choose the exercise:", exercise_options)
    MET_values = {"Biceps Curls": 3.0, "Squats": 3.5, "Side Lunges": 4.0}  # Example MET values
    MET = MET_values.get(exercise_selection, 3.0)  # Default MET value
    side = None
    report_data = {}
    if exercise_selection == "Biceps Curls":
        side = st.radio("Choose the side:", ("Left", "Right"))
        result = track_biceps_curls(side, weight, MET)
        if result:
            count, duration, calories = result
            report_data['Biceps Curls'] = {'Count': count, 'Duration': duration, 'Calories Burned': calories}
    elif exercise_selection == "Squats":
        result = track_squats(weight, MET)
        if result:
            count, duration, calories = result
            report_data['Squats'] = {'Count': count, 'Duration': duration, 'Calories Burned': calories}

    elif exercise_selection == "Side Lunges":
        result = track_lunges(weight, MET)
        if result:
            count, duration, calories = result
            report_data['Side Lunges'] = {'Count': count, 'Duration': duration, 'Calories Burned': calories}

    Gen_Report = st.button("Generate Report")
    if Gen_Report:
        report = generate_report(report_data)
        print("Debugging report_data:", report_data)
        st.write(report)



if __name__ == '__main__':
    main()