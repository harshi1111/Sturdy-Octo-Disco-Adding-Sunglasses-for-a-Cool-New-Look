# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## PROGRAM :


```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
```
# Load the Face Image
faceImage = cv2.imread(r"C:\Users\admin\Downloads\IMAGE.jpeg")
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
```
![image](https://github.com/user-attachments/assets/ef96eeaf-a8b9-4f89-91d4-d9ef0110ec8d)


```
#resized_faceImage.shape
faceImage.shape
```
![image](https://github.com/user-attachments/assets/2292981e-109d-4aa2-82c7-f41adde44ee9)


```
# Load the Sunglass image with Alpha channel
# (https://pngtree.com/freepng/red-triangle-sunglasses-black-glass_7296611.html)
glassPNG = cv2.imread(r"C:\Users\admin\Downloads\SUNGLASS.png",-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
```
![image](https://github.com/user-attachments/assets/ef5d3640-8aed-44b0-b942-d85fb49df53f)

```
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))
```
![image](https://github.com/user-attachments/assets/ae775e9f-c693-44b7-9b43-0c702b612170)


```
# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]
```

```
# Display the images for clarity
plt.figure(figsize=[20,20])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```
![image](https://github.com/user-attachments/assets/4005e0f8-37c6-4571-a165-a22526b4a309)


```
# Make a fresh copy to avoid cumulative overlays
faceWithGlassesNaive = faceImage.copy()

# Resize glasses to make them bigger
target_width = 300  # Increase width
target_height = 200  # Increase height
glassBGR = cv2.resize(glassBGR, (target_width, target_height))

# Overlay position
x1, y1 = 65, 138
x2, y2 = x1 + target_width, y1 + target_height

# Replace the eye region with the bigger sunglass image
faceWithGlassesNaive[y1:y2, x1:x2] = glassBGR

plt.imshow(faceWithGlassesNaive[..., ::-1])
plt.title("Face with Bigger Glasses")
plt.axis('off')
plt.show()
```
![image](https://github.com/user-attachments/assets/79d0a9c5-448a-4b8c-af2e-895de52bdf51)


```
# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3-channel image for the mask
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = glassMask / 255.0

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Adjust size and position (bigger + moved up)
x, y, w, h = 49, 150, 320, 160  # x = 40 → Shifted left, y = 160 → Moved up, bigger size

# Get the eye region from the face image
eyeROI = faceWithGlassesArithmetic[y:y + h, x:x + w]

# Resize glassMask and glassBGR to match eyeROI size
glassMask = cv2.resize(glassMask, (eyeROI.shape[1], eyeROI.shape[0]))
glassBGR = cv2.resize(glassBGR, (eyeROI.shape[1], eyeROI.shape[0]))

# Use float32 for better precision
maskedEye = cv2.multiply(eyeROI.astype(np.float32), (1 - glassMask.astype(np.float32)))
maskedGlass = cv2.multiply(glassBGR.astype(np.float32), glassMask.astype(np.float32))

# Combine the masked eye and glass regions
eyeRoiFinal = cv2.add(maskedEye, maskedGlass).astype(np.uint8)

# Overlay result back into the face image
faceWithGlassesArithmetic[y:y + h, x:x + w] = eyeRoiFinal

# Display results
plt.figure(figsize=[20,20])
plt.subplot(131); plt.imshow(maskedEye[...,::-1]); plt.title("Masked Eye Region")
plt.subplot(132); plt.imshow(maskedGlass[...,::-1]); plt.title("Masked Sunglass Region")
plt.subplot(133); plt.imshow(faceWithGlassesArithmetic[...,::-1]); plt.title("Augmented Face with Sunglasses")
plt.show()
```
![image](https://github.com/user-attachments/assets/136432b1-5abe-4b61-8ceb-3cba29bd7a4e)



```
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[150:150 + 160, 45:45 + 320] = eyeRoiFinal  # Using (y:y+h, x:x+w)

# Display the final result
plt.figure(figsize=[20,20])
plt.subplot(121); plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image")
plt.subplot(122); plt.imshow(faceWithGlassesArithmetic[:,:,::-1]); plt.title("With Sunglasses")
plt.show()
```
![image](https://github.com/user-attachments/assets/ca3e38d5-fc17-48aa-bf93-a0a7cde892d1)

