"""
S.E. Schimmel - S.E.Schimmel@student.tudelft.nl
Bachelor Final Project - WB3BEP-16 - TU Delft 2023-2024
Depth from Defocus using 2 differently blurred images of the same scene
"""

import Functions
import cv2
import time

start_time = time.time()

# Load the cropped image patches
sharp_image = cv2.imread(r"CROPPED-PENTLAND-DFD/50CM/50CM-F16-1.jpg")
blurred_image = cv2.imread(r'CROPPED-PENTLAND-DFD/50CM/50CM-F10-1.jpg')

height, width, channel = h, w, c = sharp_image.shape

# Convert the image patches to grayscale
gray_sharp, gray_blurred = Functions.convert_to_grayscale(sharp_image,blurred_image)

# Compute the Discrete Fourier Transform of both images
# Shift the zero-frequency component to the center of the spectrum
# Calculate the magnitude of the Fourier Transform
# Scale the magnitude for display
# Calculate difference (natural log) of images
ln_difference, scaled_magnitude_image1, scaled_magnitude_image2 = Functions.DFT_Shifted_Scaled(gray_sharp,gray_blurred)

# Calculate average of max ln difference for top 10 rows and first 10 columns
average = Functions.Calculate_Average(ln_difference,10)

# Select suitable high spatial frequencies
selected_indices = Functions.Select_Indices(ln_difference,average,10)

# Calculate Depth by taking average of multiple depth estimations
Average_Depth = Functions.Calculate_Depth(ln_difference,selected_indices, height, width)

#Print Calculated Depth
print('Calculated Depth:', Average_Depth,'mm')

#Print Runtime
print("--- %s seconds ---" % (time.time() - start_time))

# Display the magnitude of the Fourier Transform (with zero-frequency shifted to the center)
cv2.imshow('Fourier Transform Image 1', scaled_magnitude_image1)
cv2.imshow('Fourier Transform Image 2', scaled_magnitude_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()