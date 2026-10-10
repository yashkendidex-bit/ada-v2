import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "public/hand_landmarker.task"

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def main():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=python.BaseOptions.Delegate.CPU
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Hand Gesture Tracking Started...")
    print("Press 'q' to quit.")

    start_time = None

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        if start_time is None:
            start_time = cv2.getTickCount() / cv2.getTickFrequency()

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now = cv2.getTickCount() / cv2.getTickFrequency()
        ts = int((now - start_time) * 1000)
        results = landmarker.detect_for_video(mp_img, ts)

        gesture = "None"

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            lm = hand_landmarks

            index_extended = lm[8].y < lm[6].y
            middle_extended = lm[12].y < lm[10].y
            ring_extended = lm[16].y < lm[14].y
            pinky_extended = lm[20].y < lm[18].y
            thumb_extended = lm[4].x < lm[3].x if lm[5].x > lm[17].x else lm[4].x > lm[3].x

            if index_extended and middle_extended and ring_extended and pinky_extended:
                gesture = "Open Palm"
            elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture = "Closed Fist"
            elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
                tip = lm[8]
                mcp = lm[5]
                dx = tip.x - mcp.x
                dy = tip.y - mcp.y

                if abs(dy) > abs(dx):
                    gesture = "Point Up" if dy < 0 else "Point Down"
                else:
                    gesture = "Point Right" if dx > 0 else "Point Left"
            elif index_extended and middle_extended and not ring_extended and not pinky_extended:
                gesture = "Peace Sign"

            pinch_dist = get_distance(lm[4], lm[8])
            if pinch_dist < 0.05:
                gesture = "Pinching"

            cv2.putText(img, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            print(f"Detected Gesture: {gesture}")

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()