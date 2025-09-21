Drowsiness Detection

A computer vision–based system that detects driver drowsiness using facial landmarks and eye state monitoring. When the system detects signs of drowsiness, it triggers an audible alert.

🚀 Features

•Detects faces and eyes in real-time using OpenCV Haar cascades.

•Monitors eye aspect ratio to determine drowsiness.

•Plays an alarm sound when drowsiness is detected.

•Works with both images and real-time webcam video.

•Includes demo images and GIFs for testing.

📂 Project Structure

Drowsiness Detection/
│-- drowsiness_detect.py                
│-- face_and_eye_detector_single_image.py 
│-- face_and_eye_detector_webcam_video.py 
│-- requirements.txt                     
│-- shape_predictor_68_face_landmarks.dat 
│-- audio/alert.mp3                      
│-- haarcascades/                        
│-- images/                              
│-- README.md                            
│-- .gitignore

⚙️ Installation

1. Clone this repository or extract the zip:

git clone https://github.com/TejasToraskar1509/drowsiness-detection.git
cd drowsiness-detection

2. Install dependencies:

pip install -r requirements.txt

▶️ Usage

1. Run Real-time Drowsiness Detection
python drowsiness_detect.py

2. Detect Face & Eyes in an Image
python face_and_eye_detector_single_image.py

3. Detect Face & Eyes via Webcam
python face_and_eye_detector_webcam_video.py

🛠 Dependencies

•Python 3.x
•OpenCV
•dlib
•imutils
•numpy
