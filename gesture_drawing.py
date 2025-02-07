import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

shape = "None"
shapes_list = []
holding_shape = False  # Track if the shape is being held
shape_selected = False  # Ensure selection before placement
shape_size = 20  # Default shape size

# Define UI elements
buttons = {
    "Circle": (180, 10, 230, 50),
    "Square": (280, 10, 330, 50),
    "Triangle": (380, 10, 430, 50),
}
clear_button = (500, 10, 580, 50)
held_shape_position = None  # Position of hovering shape

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw selection buttons
    for btn_text, (x1, y1, x2, y2) in buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
        cv2.putText(frame, btn_text[0], (x1 + 15, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Draw "Clear" button
    cv2.rectangle(frame, (clear_button[0], clear_button[1]), (clear_button[2], clear_button[3]), (200, 50, 50), -1)
    cv2.putText(frame, "Clear", (clear_button[0] + 15, clear_button[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)
        right_hand = results.multi_hand_landmarks[0]  # First detected hand (assumed right)
        left_hand = results.multi_hand_landmarks[1] if hand_count > 1 else None  # Second hand if detected (left)

        mp_drawing.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)

        # Get right-hand fingers (holding shape)
        index_finger = right_hand.landmark[8]
        middle_finger = right_hand.landmark[12]
        ix, iy = int(index_finger.x * w), int(index_finger.y * h)
        mx, my = int(middle_finger.x * w), int(middle_finger.y * h)

        # Check if right-hand index finger is inside any button
        for btn_text, (x1, y1, x2, y2) in buttons.items():
            if x1 < ix < x2 and y1 < iy < y2:
                shape = btn_text  # Update selected shape
                shape_selected = True  # Mark shape as selected
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), -1)
                cv2.putText(frame, btn_text[0], (x1 + 15, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Check if right-hand index finger is over the "Clear" button
        if clear_button[0] < ix < clear_button[2] and clear_button[1] < iy < clear_button[3]:
            shapes_list.clear()  # Clear all placed shapes
            cv2.rectangle(frame, (clear_button[0], clear_button[1]), (clear_button[2], clear_button[3]), (0, 200, 0), -1)
            cv2.putText(frame, "Clear", (clear_button[0] + 15, clear_button[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Detect if right-hand fingers are touching (holding shape)
        finger_distance = abs(ix - mx) + abs(iy - my)
        if finger_distance < 20 and shape_selected:
            holding_shape = True
            held_shape_position = (ix, iy)  # Move shape with fingers

        elif holding_shape and finger_distance > 30:
            # Release and deploy shape
            shapes_list.append((shape, held_shape_position, shape_size))
            holding_shape = False
            shape_selected = False
            held_shape_position = None
            shape = "None"

        # Detect left-hand pinch for resizing
        if left_hand:
            mp_drawing.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)
            thumb_tip = left_hand.landmark[4]
            index_tip = left_hand.landmark[8]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            pinch_distance = np.linalg.norm([thumb_x - index_x, thumb_y - index_y])

            # Scale size dynamically based on pinch distance
            shape_size = int(pinch_distance / 2)  # Adjust scaling factor as needed
            shape_size = max(10, min(shape_size, 100))  # Keep size within reasonable limits

    # Draw placed shapes
    for shape_type, pos, size in shapes_list:
        if shape_type == "Circle":
            cv2.circle(frame, pos, size, (255, 255, 255), -1)
        elif shape_type == "Square":
            cv2.rectangle(frame, (pos[0] - size, pos[1] - size),
                          (pos[0] + size, pos[1] + size), (255, 255, 255), -1)
        elif shape_type == "Triangle":
            pts = np.array([[pos[0], pos[1] - size],
                            [pos[0] - size, pos[1] + size],
                            [pos[0] + size, pos[1] + size]], np.int32)
            cv2.fillPoly(frame, [pts], (255, 255, 255))

    # Draw hovering shape (if holding)
    if holding_shape and held_shape_position:
        if shape == "Circle":
            cv2.circle(frame, held_shape_position, shape_size, (0, 255, 255), 2)
        elif shape == "Square":
            cv2.rectangle(frame, (held_shape_position[0] - shape_size, held_shape_position[1] - shape_size),
                          (held_shape_position[0] + shape_size, held_shape_position[1] + shape_size), (0, 255, 255), 2)
        elif shape == "Triangle":
            pts = np.array([[held_shape_position[0], held_shape_position[1] - shape_size],
                            [held_shape_position[0] - shape_size, held_shape_position[1] + shape_size],
                            [held_shape_position[0] + shape_size, held_shape_position[1] + shape_size]], np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

    cv2.imshow("Resize with Pinch", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
