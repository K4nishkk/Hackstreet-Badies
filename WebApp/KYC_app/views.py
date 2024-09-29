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
import os
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

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


# Keep track of the last 8 frames' results for liveness detection
last_10_frames = []

# Path to save captured images
IMAGE_SAVE_PATH = 'KYC_app/static/captured_images/'  # Adjust the path as needed

# Ensure the save path directory exists
if not os.path.exists(IMAGE_SAVE_PATH):
    os.makedirs(IMAGE_SAVE_PATH)


def process_image(image_data):
    try:
        # Decode the base64 image data
        image_data = image_data.split(',')[1]  # Extract base64 part of the image data URL
        # print(image_data)
        image_data = base64.b64decode(image_data)  # Decode the base64 to binary

        # Convert the binary image data to a PIL Image
        image = Image.open(BytesIO(image_data))
        # print(image)
        image_np = np.array(image)

        # Example placeholder for the image processing
        # Convert RGB to BGR format for OpenCV
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # print(frame)

        # Perform face and blink detection (you should replace this with your actual detection logic)
        faces = detector(frame)
        # print(faces)
        if len(faces) == 0:
            return False, "No face detected.", None

        for face in faces:
            shape = predictor(frame, face)
            # print(shape)
            shape_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)

            blink_detected = check_blink(shape_np)
            if blink_detected:
                return True, "Blink detected, person is alive.", image

        return False, "No blink detected, might be a spoof.", None
    except Exception as e:
        return False, f"Error processing image: {str(e)}", None


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
    global last_10_frames

    if request.method == 'POST':
        try:
            # Decode the incoming JSON body
            body = json.loads(request.body.decode('utf-8'))
            image_data = body.get('image_data')

            # Process the frame for liveness detection
            is_live, message, image = process_image(image_data)

            # Update the list of last 10 frames
            if len(last_10_frames) >= 10:
                last_10_frames.pop(0)  # Remove the oldest frame status
            last_10_frames.append(is_live)  # Add the current frame's liveness status

            # Initialize image filename as None
            image_filename = None

            # Check if 2 out of the last 10 frames are live
            if last_10_frames.count(True) >= 2:
                # Save the image
                image_filename = f"captured_image_{len(os.listdir(IMAGE_SAVE_PATH)) + 1}.png"
                image_save_path = os.path.join(IMAGE_SAVE_PATH, image_filename)
                image.save(image_save_path)

                # Clear the frame history after capturing
                last_10_frames = []

                # Append to the message that an image was saved
                message += f" Image saved as {image_filename}."

            # Create response message
            response_message = "Liveness Detected: " + str(is_live) + ". " + message

            # Log the response message for debugging
            print(response_message)

            # Return the response with the image URL if an image was saved
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

@csrf_exempt
def upload_document(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        
        # Set a save path (e.g., to media directory)
        save_path = os.path.join('media', uploaded_file.name)

        # Save the file to the specified location
        path = default_storage.save(save_path, ContentFile(uploaded_file.read()))
        
        return JsonResponse({'message': 'File uploaded successfully!', 'path': path})
    return JsonResponse({'error': 'Invalid request or no file provided.'}, status=400)