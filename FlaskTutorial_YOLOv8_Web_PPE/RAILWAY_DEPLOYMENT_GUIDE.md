# Railway Deployment Guide

This guide will help you properly deploy your YOLOv8 Object Detection application to Railway with cloud mode enabled.

## 1. Setting Environment Variables in Railway

The application needs the `CLOUD_MODE` environment variable set to `true` to properly function in the cloud environment:

1. Log in to your [Railway dashboard](https://railway.app/dashboard)
2. Select your Object Detection project
3. Go to the "Variables" tab
4. Click "New Variable"
5. Add the following variable:
   - **Name**: `CLOUD_MODE`
   - **Value**: `true`
6. Click "Add" to save the variable

## 2. Add Demo Video to Your Repository

For the cloud mode to work properly, you need to add a demo video to your repository:

1. Download a small demo video (from Pexels.com or another free source)
2. Name it `demo.mp4`
3. Place it in the `static/files/` directory in your project
4. Commit and push this change to your repository

## 3. Push Updated Code to Railway

Ensure your latest code with all the cloud detection improvements is deployed:

1. Make sure all your changes are committed:
   ```bash
   git add .
   git commit -m "Update code for Railway deployment with cloud mode"
   ```

2. If you're using GitHub as your source in Railway:
   ```bash
   git push origin main
   ```
   Railway will automatically deploy the new code.

3. If you're deploying directly to Railway:
   ```bash
   railway up
   ```
   (You need the Railway CLI installed for this command)

## 4. Verify Your Deployment

1. Once deployed, visit your Railway app URL (e.g., https://web-production-xxxx.up.railway.app/)
2. Navigate to the "LiveWebcam" section
3. You should see:
   - A cloud mode notification banner
   - The demo video playing with object detection applied to it
   - No webcam permission requests (since it's using the demo video)

## Troubleshooting

If you're still seeing issues with the application on Railway:

1. **Check Railway Logs**:
   - Go to your project in the Railway dashboard
   - Click on the "Deployments" tab
   - Click on the latest deployment
   - Click "View Logs"
   - Look for any error messages or issues

2. **Verify Demo Video**:
   - Make sure your demo video is properly committed and pushed
   - Check the file exists in the repository
   - Ensure it's in one of these paths: `/static/demo.mp4`, `/static/demo_video.mp4`, or `/static/files/demo.mp4`

3. **Environment Variables**:
   - Double-check that `CLOUD_MODE=true` is set in your Railway variables

4. **Filesystem Access**:
   - Note that Railway has ephemeral storage, which means uploaded files won't persist
   - Ensure your app is using the demo video from the repository, not expecting uploaded files to persist 