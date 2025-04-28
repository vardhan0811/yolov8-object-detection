# Bypassing Self-Signed Certificate Warnings in Browsers

When running the application with HTTPS locally, you'll encounter certificate warnings because we're using a self-signed certificate. This is normal and expected for development environments.

Here's how to bypass these warnings in different browsers:

## Google Chrome / Microsoft Edge

### Method 1: Advanced Option (Visible UI)
1. Click on the "Not secure" or warning message
2. Click on "Advanced"
3. Click on "Proceed to localhost (unsafe)" or similar text

### Method 2: Keyboard Shortcut (Hidden Feature)
1. Click anywhere on the error page
2. Type exactly: `thisisunsafe` (you won't see the text appear)
3. The page will immediately load, bypassing the warning

## Firefox

### Method 1: Add Exception
1. Click on "Advanced" button
2. Click on "Accept the Risk and Continue"

## Safari

### Method 1: Proceed Anyway
1. Click on "Show Details"
2. Click on "visit this website"
3. Click on "Visit Website" in the confirmation dialog
4. Enter your computer password if prompted

## Internet Explorer / Edge Legacy

### Method 1: Continue to Website
1. Click on "More information"
2. Click on "Go on to the webpage (not recommended)"

## Why These Warnings Appear

These warnings appear because:
1. The certificate is self-signed (not issued by a trusted certificate authority)
2. The browser can't verify the identity of the website
3. This is expected and normal for development environments

## Security Considerations

- These bypasses are safe when you're accessing a local development server that you control
- Never bypass certificate warnings for real websites on the internet
- For production deployments, always use proper SSL certificates from trusted certificate authorities

## Alternative: Running Without HTTPS

If you continue to have trouble with certificate warnings, you can run the application without HTTPS:

```bash
# Simply run without the USE_HTTPS environment variable
python flaskapp.py
```

However, note that without HTTPS, **webcam access will not work in modern browsers**. This is a security restriction imposed by browsers to protect users. 