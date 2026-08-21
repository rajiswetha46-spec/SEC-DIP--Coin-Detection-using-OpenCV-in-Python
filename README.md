# SEC-DIP--Coin-Detection-using-OpenCV-in-Python
### Name: SWETHA R
### Reg No: 212225100055
## Aim:
To detect and count the number of coins in an image using OpenCV by applying grayscale conversion, thresholding, morphological operations, blob detection, and contour detection.
## Algorithm:
1.Import the required OpenCV, NumPy, and Matplotlib libraries.
2.Read and display the input coin image.
3.Convert the input image from BGR to grayscale.
4.Apply thresholding to separate the coins from the background.
5.Perform morphological Opening to remove small noise.
6.Perform morphological Closing to fill small gaps and improve the coin regions.
7.Apply Blob Detection to detect coin-like objects and count them.
8.Apply Contour Detection to identify the boundaries of the coins and count them.
9.Display the detected coins and their boundaries.
10.Compare the results obtained using Blob Detection and Contour Detection.
## Program

### Original Image:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("coin.png")

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()
```
<img width="399" height="635" alt="image" src="https://github.com/user-attachments/assets/397cecfe-b01e-47c9-b996-4797936421e7" />

### Grayscale Image:

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 6))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
```

<img width="392" height="640" alt="image" src="https://github.com/user-attachments/assets/5801a77a-641f-4007-907d-127fca182e75" />

### Thresholding:

```
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(8, 6))
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()
```
<img width="405" height="645" alt="image" src="https://github.com/user-attachments/assets/03d6b467-1c8d-4cc7-948a-e4f4318ebb83" />

### Morphological Opening & Closing
```
kernel = np.ones((5, 5), np.uint8)

# Opening - removes small noise
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Closing - fills small holes
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(opening, cmap="gray")
plt.title("After Opening")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(closing, cmap="gray")
plt.title("After Closing")
plt.axis("off")

plt.show()
```
<img width="975" height="537" alt="image" src="https://github.com/user-attachments/assets/bd309299-fcb6-412c-8a15-22e25e139f89" />

### Blob Detection:

```
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 500
params.maxArea = 50000

params.filterByCircularity = True
params.minCircularity = 0.7

params.filterByConvexity = True
params.minConvexity = 0.8

params.filterByInertia = True
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(closing)

blob_image = cv2.drawKeypoints(
    cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR),
    keypoints,
    None,
    (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(blob_image, cv2.COLOR_BGR2RGB))
plt.title("Blob Detection")
plt.axis("off")
plt.show()

print("Number of coins detected using Blob Detection:", len(keypoints))
```
<img width="409" height="635" alt="image" src="https://github.com/user-attachments/assets/1293c00b-8bae-483d-8d38-ba56b09bddba" />


### Contour Detection
```
contours, hierarchy = cv2.findContours(
    closing,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Filter small contours
coin_contours = []

for contour in contours:
    area = cv2.contourArea(contour)
    
    if area > 500:
        coin_contours.append(contour)

# Draw contours
contour_image = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

cv2.drawContours(
    contour_image,
    coin_contours,
    -1,
    (0, 255, 0),
    2
)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title("Contour Detection")
plt.axis("off")
plt.show()

print("Number of coins detected using Contours:", len(coin_contours))
```
<img width="402" height="621" alt="image" src="https://github.com/user-attachments/assets/70a9e097-b7f3-459c-b450-d07dd6668a64" />


### Coin counting

```
print("===== RESULTS =====")
print("Coins detected using Blob Detection    :", len(keypoints))
print("Coins detected using Contour Detection :", len(coin_contours))
```
<img width="460" height="90" alt="image" src="https://github.com/user-attachments/assets/a16e8673-6922-405d-915f-7dd1b3325f89" />


### Discussion and conclusion:
<img width="1223" height="414" alt="image" src="https://github.com/user-attachments/assets/e275a629-53c7-414c-8fb9-426c018c4d3c" />

## Result:

The coins in the input image were successfully detected and counted using both Blob Detection and Contour Detection. The thresholding and morphological operations helped to improve the image quality and separate the coins from the background.

The number of coins detected using Blob Detection was obtained from the detected keypoints, while the number of coins detected using Contour Detection was obtained from the filtered contours.
