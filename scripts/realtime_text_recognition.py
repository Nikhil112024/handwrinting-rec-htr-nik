import cv2
import numpy as np
import onnxruntime as ort
import json
import time

# ------------------ Load ONNX Models (GPU) ------------------
detector_model_path = "../htr_pipeline/models/detector.onnx"
reader_model_path = "../htr_pipeline/models/reader.onnx"
reader_metadata_path = "../htr_pipeline/models/reader.json"

detector_session = ort.InferenceSession(detector_model_path, providers=["CUDAExecutionProvider"])
reader_session = ort.InferenceSession(reader_model_path, providers=["CUDAExecutionProvider"])

# Load character mapping
with open(reader_metadata_path, "r") as f:
    metadata = json.load(f)

char_list = metadata.get("char_list", [])
if not char_list:
    char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-()'\" "


# ------------------ Preprocess Image ------------------
def preprocess_image(image):
    """ Converts image to grayscale, enhances contrast, and normalizes it """
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    gray = cv2.equalizeHist(gray)

    # Adaptive Thresholding for better text segmentation
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

    return gray


# ------------------ Detect and Extract Text Regions ------------------
def extract_text_regions(image):
    """ Finds bounding boxes for text regions using OpenCV contours """
    if image is None:
        return []

    processed_img = preprocess_image(image)

    # Detect edges
    edges = cv2.Canny(processed_img, 50, 150)

    # Find contours (bounding boxes for words/lines)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store text regions
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 40 and h > 15:  # Ignore small noise
            text_regions.append((x, y, w, h))

    # Sort text regions from top to bottom, left to right
    text_regions = sorted(text_regions, key=lambda b: (b[1], b[0]))

    print(f"🔍 Detected {len(text_regions)} text regions.")

    return text_regions


# ------------------ Decode Model Output ------------------
def decode_output(softmax_output, char_list):
    """ Converts softmax probabilities into text """
    text = ""
    for timestep in softmax_output:
        max_index = np.argmax(timestep)  # Get most probable character index
        if max_index < len(char_list):  # Avoid index out of bounds
            text += char_list[max_index]
    return text.strip()


# ------------------ Process Captured Image ------------------
def process_captured_image(image):
    """ Runs detection and recognition on each detected text region """
    if image is None:
        return "⚠ No Image Captured!"

    text_regions = extract_text_regions(image)

    if not text_regions:
        return "⚠ No Text Detected!"

    full_text = ""
    for (x, y, w, h) in text_regions:
        roi = image[y:y + h, x:x + w]  # Extract region of interest (ROI)

        # Preprocess ROI for ONNX model
        try:
            roi = cv2.resize(roi, (128, 32))  # Resize for model input
            roi = roi.astype(np.float32) / 255.0  # Normalize
            roi = np.expand_dims(roi, axis=[0, 1])  # Add batch and channel dimensions

            print(f"🔹 Processing text region at ({x},{y},{w},{h})")

            # Run Reader Model
            reader_output = reader_session.run(None, {reader_session.get_inputs()[0].name: roi})

            # Decode Softmax Output into full text
            recognized_text = decode_output(reader_output[0][0], char_list)

            print(f"✅ Recognized Text: {recognized_text}")

            # Append detected text
            full_text += recognized_text + " "

        except Exception as e:
            print(f"⚠ Model Processing Error: {str(e)}")
            return "⚠ Error in Text Recognition!"

    return full_text.strip()


# ------------------ Start Webcam Capture ------------------
cap = cv2.VideoCapture(0)  # Open default camera
captured_frame = None
recognized_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame!")
        break

    # Resize frame for display
    frame_resized = cv2.resize(frame, (640, 480))

    # Create a blank canvas for displaying extracted text
    text_display = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background

    # Add extracted full document text to right panel
    y_offset = 50
    for line in recognized_text.split("\n"):
        cv2.putText(text_display, line, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_offset += 30

    # If no text is detected, show a message
    if recognized_text.strip() == "":
        cv2.putText(text_display, "⚠ No Text Detected!", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw bounding boxes on the captured image (for debugging)
    if captured_frame is not None:
        text_regions = extract_text_regions(captured_frame)
        for (x, y, w, h) in text_regions:
            cv2.rectangle(captured_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Concatenate the two frames side by side
    if captured_frame is not None:
        left_display = cv2.resize(captured_frame, (640, 480))  # Left Side → Captured Image
    else:
        left_display = frame_resized  # If no capture, show live feed

    final_display = np.hstack((left_display, text_display))

    # Show the split-screen UI
    cv2.imshow("Real-time Document OCR | Left: Image | Right: Extracted Text", final_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        print("📸 Capturing Image...")
        captured_frame = frame.copy()  # Save captured frame
        filename = f"captured_{int(time.time())}.jpg"
        cv2.imwrite(filename, captured_frame)
        print(f"✅ Image saved as {filename}")

        # Process Image and Get Recognized Full Document Text
        recognized_text = process_captured_image(captured_frame)
        print(f"📝 Recognized Document Text:\n{recognized_text}")

    elif key == ord('q'):
        print("Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
