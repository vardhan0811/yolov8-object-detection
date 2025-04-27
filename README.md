# YOLOv8 Object Detection System

This application uses YOLOv8 models for real-time object detection. It provides two models:
1. PPE Detection - for detecting personal protective equipment (helmets, gloves, etc.)
2. General Object Detection - for detecting common objects (people, cars, books, etc.)

## Features

- Upload and process video files
- Real-time webcam object detection
- Toggle between PPE detection and general object detection modes
- Visual bounding boxes with labels for detected objects

## Deployment Instructions

### Local Deployment

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
cd FlaskTutorial_YOLOv8_Web_PPE
python flaskapp.py
```

4. Open the application in your browser at `http://localhost:5000`

### Deploying to Railway

1. Create a Railway account at [railway.app](https://railway.app) if you don't have one

2. Install the Railway CLI:
```bash
npm i -g @railway/cli
```

3. Login to Railway:
```bash
railway login
```

4. Initialize your project:
```bash
railway init
```

5. Link your repository to Railway:
```bash
railway link
```

6. Deploy your application:
```bash
railway up
```

7. Open your deployed application:
```bash
railway open
```

## Important Notes for Railway Deployment

- Webcam functionality is disabled in cloud deployment as browser webcam access requires HTTPS and server-side webcam access is not available in cloud environments
- Use the "Upload Video" feature to test object detection in the cloud environment
- The application will automatically download model weights if they're not included in the deployment

## Models

The application uses two YOLOv8 models:

1. **ppe.pt** - A custom model trained for PPE detection with 7 classes:
   - Protective Helmet
   - Shield
   - Jacket
   - Dust Mask
   - Eye Wear
   - Glove
   - Protective Boots

2. **yolov8n.pt** - The standard YOLOv8 nano model with 80 classes from the COCO dataset

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Flask](https://flask.palletsprojects.com/) for the web framework 