import cv2
import numpy as np
import base64
import dlib
from io import BytesIO
from PIL import Image
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)


def index(request):
    return render(request, 'KYC_app/index.html')

# def process_image(request):
#     if request.method == 'POST':
#         data = request.body.decode('utf-8')
#         image_data = base64.b64decode(data.split(',')[1])  # Extract the base64 part

#         # Optional: Save the image to file or process it
#         image = Image.open(BytesIO(image_data))
#         image.save('captured_image.png')  # Save to a file for later use

#         return JsonResponse({'message': 'Image processed successfully'})
#     else:
#         return JsonResponse({'message': 'Invalid request'}, status=400)





# Initialize dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("KYC_app/models/shape_predictor_68_face_landmarks.dat")

# Indices for the left and right eye landmarks
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

def process_image(image_data):
    try:
        # Decode the base64 image data
        image_data = image_data.split(',')[1]  # Extract base64 part of the image data URL
        image_data = base64.b64decode(image_data)  # Decode the base64 to binary

        # Convert the binary image data to a PIL Image
        image = Image.open(BytesIO(image_data))
        image_np = np.array(image)

        # Example placeholder for the image processing
        # Convert RGB to BGR format for OpenCV
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Perform face and blink detection (you should replace this with your actual detection logic)
        faces = detector(frame)
        if len(faces) == 0:
            return False, "No face detected."

        for face in faces:
            shape = predictor(frame, face)
            shape_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)

            blink_detected = check_blink(shape_np)
            if blink_detected:
                return True, "Blink detected, person is alive."

        return False, "No blink detected, might be a spoof."
    except Exception as e:
        return False, f"Error processing image: {str(e)}"


def check_blink(shape):
    # Calculate the Eye Aspect Ratio (EAR) to detect blinks
    def eye_aspect_ratio(eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    left_eye = shape[LEFT_EYE_POINTS]
    right_eye = shape[RIGHT_EYE_POINTS]
    
    # Compute the eye aspect ratio (EAR) for both eyes
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    
    # Average EAR
    ear = (left_ear + right_ear) / 2.0
    
    # If EAR is below a threshold, it indicates a blink
    if ear < 0.25:  # Threshold for blink detection
        return True
    return False

def liveness_detection(request):
    if request.method == 'POST':
        try:
            # Decode the incoming JSON body
            body = json.loads(request.body.decode('utf-8'))
            image_data = body.get('image_data')

            # Process the frame for liveness detection
            is_live, message = process_image(image_data)

            # Create response message
            response_message = "Liveness Detected: " + str(is_live) + ". " + message
            
            # Log the response message for debugging
            # logger.info(response_message)
            print(response_message)

            # Return plain text response
            return HttpResponse(response_message, content_type='text/plain')

        except json.JSONDecodeError:
            error_message = 'Invalid JSON provided.'
            logger.error(error_message)
            return HttpResponse(error_message, status=400, content_type='text/plain')
        except Exception as e:
            error_message = f'Error: {str(e)}'
            logger.error(error_message)
            return HttpResponse(error_message, status=500, content_type='text/plain')

    return HttpResponse('Invalid request method.', status=400, content_type='text/plain')