from ultralytics import YOLO
import cv2
import os

def test_detection():
    # Path to the test image
    image_path = "Images/image1.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Path to the models
    ppe_model_path = "YOLO-Weights/ppe.pt"
    general_model_path = "yolov8n.pt"
    
    # Choose which model to use
    model_path = general_model_path  # Change to ppe_model_path for PPE detection
    
    print(f"Loading model from: {model_path}")
    
    # Load the model
    model = YOLO(model_path)
    
    # Read the image
    img = cv2.imread(image_path)
    
    # Run detection
    results = model(img)
    
    # Process and display the results
    for r in results:
        img_with_boxes = r.plot()
        
        # Save the output image
        output_path = "detection_output.jpg"
        cv2.imwrite(output_path, img_with_boxes)
        print(f"Detection completed. Output saved to: {output_path}")
        
        # Display the image (only works with GUI)
        try:
            cv2.imshow("Detection Result", img_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Could not display image (no GUI available)")

if __name__ == "__main__":
    test_detection() 