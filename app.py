import cv2
from roboflow import Roboflow
rf = Roboflow(api_key="################")
project = rf.workspace().project("football-players-xgy1l")
model = project.version(2).model


# Path to the video file
video_path = 'video (2160p).mp4'

# Open the video file
video = cv2.VideoCapture(video_path)

# Read the first frame to get the dimensions
ret, frame = video.read()
frame = cv2.resize(frame, (640, 640))
height, width, _ = frame.shape

# Specify the output video path
output_path = 'output_video.mp4'

# Create a VideoWriter object to save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = video.get(cv2.CAP_PROP_FPS)  # Get the frames per second (fps) of the original video
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Read and process frames until the video ends
while video.isOpened():
    ret, frame = video.read()

    # Break the loop if no more frames are available
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))

    pred = model.predict(frame, confidence=10, overlap=10).json()  
    predictions = pred['predictions']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
    for bounding_box in predictions:
        x0 = int(bounding_box['x']) - int(bounding_box['width'] )/ 2
        x1 = int(bounding_box['x']) + int(bounding_box['width'] )/ 2
        y0 = int(bounding_box['y']) - int(bounding_box['height']) / 2
        y1 = int(bounding_box['y']) + int(bounding_box['height']) / 2

        class_label = bounding_box['class']
        confidence = bounding_box['confidence']
        label = f"{class_label} {confidence:.2f}"
        
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(frame, start_point, end_point, color=(0,255,0), thickness=2)
        cv2.putText(frame, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 1, 146), 2)

    # Write the annotated frame to the output video
    output_video.write(frame)

    # Display the frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects and close windows
video.release()
output_video.release()
cv2.destroyAllWindows()
