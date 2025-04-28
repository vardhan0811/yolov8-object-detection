"""
Setup script to install necessary dependencies for HTTPS support
"""
import sys
import subprocess
import os

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("Setting up HTTPS dependencies...")
    
    # List of required packages
    packages = ["cryptography"]
    
    for package in packages:
        try:
            install_package(package)
            print(f"Successfully installed {package}")
        except Exception as e:
            print(f"Error installing {package}: {e}")
            return False
    
    print("\nAll dependencies installed successfully!")
    print("\nTo run the application with HTTPS:")
    
    if os.name == 'nt':  # Windows
        print("\nWindows:")
        print("$env:USE_HTTPS=\"true\"")
        print("python flaskapp.py")
    else:  # Unix/Linux/Mac
        print("\nLinux/Mac:")
        print("export USE_HTTPS=true")
        print("python flaskapp.py")
    
    print("\nThen visit: https://localhost:5000")
    print("\nNote: You'll need to accept the security warning in your browser since this uses a self-signed certificate.")
    
    return True

if __name__ == "__main__":
    main() 