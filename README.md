# YOLOv8-Door-detection-for-visually-impaired-people


# YOLOv8 Door Detection for Visually Impaired Individuals

Imagine a world where doors become more than just obstacles for the visually impaired. Our YOLOv8 project revolutionizes accessibility by using cutting-edge computer vision technology to detect doors.

## Dataset

The dataset has been collected from New York and annotated through a combination of auto-annotation and manual enhancement.

**Dataset Download:**
[Download Dataset](https://drive.google.com/file/d/1-0dWfmeUXN7V1tvZQubcRW6frCEu8fjq/view?usp=sharing)

## Installation and Usage

1. Install the required library:

```bash
   pip install ultralytics
   ```

2. Load the model:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8s.pt")  # Load the pretrained model
   ```

3. Training the model:
   ```python
   model.train(data='/content/data.yaml', epochs=47, imgsz=1280, batch=8)
   ```

4. Load the best model:
   ```python
   model = YOLO("/content/runs/detect/train2/weights/best.pt")  # Load the best model
   ```

5. Making predictions:
   ```python
   res = model.predict("/content/test.png", save=True, conf=0.3)
   ```

6. Visualizing results:
   ```python
   import matplotlib.pyplot as plt
   import matplotlib.image as mpimg

   # Load your images
   image1 = mpimg.imread('/content/test.png')
   image2 = mpimg.imread('/content/runs/detect/predict2/test.png')

   # Plotting the images side by side
   plt.figure(figsize=(20, 20))
   plt.subplot(1, 2, 1)
   plt.imshow(image1)
   plt.title('Source Image')

   plt.subplot(1, 2, 2)
   plt.imshow(image2)
   plt.title('Predicted Image')

   plt.tight_layout()
   plt.show()
   ```

### upcoming 

deployment on hugging face
