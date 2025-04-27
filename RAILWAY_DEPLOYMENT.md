# Railway Deployment Guide for YOLOv8 Object Detection

This guide explains how to deploy the YOLOv8 Object Detection application to Railway and ensure that detection functionality works correctly.

## Prerequisites

- A [Railway](https://railway.app/) account
- Git repository with this code
- Railway CLI (optional)

## Deployment Steps

### 1. Preparing Your Repository

Ensure your repository has the following files properly configured:

- `requirements.txt` - Contains all Python dependencies
- `railway.json` - Contains system dependencies and deployment configuration
- `Procfile` - Specifies how to run the application
- `.env.railway` - Environment variables specific to Railway

### 2. Deploy to Railway

#### Option 1: Deploy via Railway Dashboard

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically detect the configuration from `railway.json`

#### Option 2: Deploy via Railway CLI

```bash
# Login to Railway
railway login

# Initialize a new project
railway init

# Link to your existing project (if you've already created one)
railway link

# Deploy your application
railway up
```

### 3. Environment Variables

Make sure to set these environment variables in your Railway project settings:

- `RAILWAY_ENVIRONMENT=true`
- `YOLO_DEVICE=cpu`
- `YOLO_VERBOSE=False`

You can import them all from the `.env.railway` file.

### 4. Files Required for Detection

Make sure these files are included in your Git repository before deployment:

- YOLO-Weights/ppe.pt
- yolov8n.pt

If the model files are too large for Git, configure Railway to download them during deployment by adding this to your `Procfile`:

```
web: mkdir -p YOLO-Weights && wget -nc -O YOLO-Weights/ppe.pt https://your-storage-url/ppe.pt && wget -nc -O yolov8n.pt https://your-storage-url/yolov8n.pt && gunicorn enhanced:app --bind=0.0.0.0:$PORT
```

Replace `https://your-storage-url/` with the actual URL where you've stored the model files.

### 5. Memory and CPU Allocation

In Railway dashboard:

1. Go to your project settings
2. Increase memory allocation to at least 1GB (2GB recommended)
3. Set CPU allocation to at least 1 vCPU

### 6. Webcam Limitations

Note that webcam functionality will NOT work in Railway or any cloud deployment. The application automatically disables webcam features in cloud environments.

Your users will still be able to:
- Upload videos for processing
- Use pre-recorded demo videos
- View detection results on images

### 7. Testing Deployment

After deployment, test the following features:

1. Open the app URL provided by Railway
2. Upload a video file for detection
3. Check the detection results

### 8. Troubleshooting

If detection is not working:

1. Check Railway logs for errors
2. Ensure model files are available in the deployment
3. Verify system dependencies are correctly specified in `railway.json`
4. Increase memory allocation if seeing OOM (out of memory) errors

## Advanced Configuration

For further optimization:

- Enable persistent storage on Railway for model caching
- Configure AWS S3 or similar for storing uploaded and processed videos
- Implement a queue system for processing larger videos asynchronously