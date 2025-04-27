from ultralytics import YOLO
import cv2
import os

# Get correct paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Original COCO model (80 classes)
coco_model_path = os.path.join(project_root, "YOLO-Weights", "yolov8n.pt")

# Open Images V7 model (600 classes)
oiv7_model_path = os.path.join(project_root, "YOLO-Weights", "yolov8n-oiv7.pt")

image_path = os.path.join(project_root, "Images", "2.png")

# Check if model exists, if not download it
if not os.path.exists(oiv7_model_path):
    print(f"Downloading Open Images V7 model with 600 classes...")
    oiv7_model = YOLO("yolov8n-oiv7.pt")  # This will download the model
    oiv7_model.save(oiv7_model_path)
else:
    oiv7_model = YOLO(oiv7_model_path)

print(f"Loading model from: {oiv7_model_path}")
print(f"Model loaded with {len(oiv7_model.names)} classes")
print(f"Processing image: {image_path}")

# Run detection with Open Images model
results = oiv7_model(image_path, show=True)

# Display detected objects
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = oiv7_model.names[cls]
        print(f"Detected: {class_name} with confidence {conf:.2f}")

print("Press any key to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
