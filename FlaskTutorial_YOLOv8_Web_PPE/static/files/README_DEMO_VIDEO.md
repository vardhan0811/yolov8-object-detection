# Demo Video for Cloud Deployment

For cloud deployments like Railway, this directory should contain a demo video file that will be used when webcam access is limited.

## Requirements:
1. Name the file `demo.mp4`
2. Keep the file relatively small (1-5 MB if possible)
3. Choose content that will demonstrate object detection well (with people, cars, etc.)

## Where to get a demo video:
- Download from Pexels.com or other free stock video sites
- Use a short clip from the YOLOv8 repository: https://media.githubusercontent.com/media/ultralytics/assets/main/yolov8_video.mp4
- Record your own short video showing common objects

## Manual upload:
If you're deploying to Railway, you need to:
1. Download a demo video locally
2. Name it `demo.mp4` 
3. Place it in the `static/files/` directory
4. Commit and push the change to your repository
5. Redeploy your application

Having this file is crucial for the cloud detection mode to work properly. 