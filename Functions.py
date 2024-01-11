"""
S.E. Schimmel - S.E.Schimmel@student.tudelft.nl
Bachelor Final Project - WB3BEP-16 - TU Delft 2023-2024
Depth from Defocus using a sharp and blurred image of the same scene
"""

import cv2
import numpy as np

# Known camera intrinsic values
F=55 #mm (= Focal Length of Camera Lens)
v0=59.06 #mm (Focus Distance = 80cm)
f1=5.6 # (= f-number at which the lens aperture is set)

# Convert image to grayscale image (also changes image shape from height, width, channel to height, width only)
def convert_to_grayscale(sharp_image,blurred_image):
    gray_sharp = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    return(gray_sharp,gray_blurred)

# Compute the Discrete Fourier Tranform, Shift the zero-frequency component, Compute the magnitude and Scale the magnitude for display
def DFT_Shifted_Scaled(gray_sharp,gray_blurred):
    # Compute the discrete Fourier Transform of the images
    fourier_sharp = cv2.dft(np.float32(gray_sharp), flags=cv2.DFT_COMPLEX_OUTPUT)
    fourier_blurred = cv2.dft(np.float32(gray_blurred), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift_sharp = np.fft.fftshift(fourier_sharp)
    fourier_shift_blurred = np.fft.fftshift(fourier_blurred)

    # Calculate the magnitude of the Fourier Transform
    magnitude_sharp = 20 * np.log(cv2.magnitude(fourier_shift_sharp[:, :, 0], fourier_shift_sharp[:, :, 1]))
    magnitude_blurred = 20 * np.log(cv2.magnitude(fourier_shift_blurred[:, :, 0], fourier_shift_blurred[:, :, 1]))

    # Scale the magnitude for display
    scaled_magnitude_image1 = cv2.normalize(magnitude_sharp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    scaled_magnitude_image2 = cv2.normalize(magnitude_blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Calculate difference between natural log of images
    ln_FT_sharp = np.log(magnitude_sharp)
    ln_FT_blurred = np.log(magnitude_blurred)
    ln_difference = ln_FT_sharp - ln_FT_blurred

    return(ln_difference,scaled_magnitude_image1,scaled_magnitude_image2)

# Calculate the average of the maximum values in terms of ln_difference from the top 10 rows and most left 10 columns
def Calculate_Average(ln_difference,n):
    sum = 0
    for x in range(n):
        sum += np.max(ln_difference[x])
        sum += np.max(ln_difference[:, x])
    average = sum / (2 * n)
    return(average)

#  Save the indices from suitable frequencies
def Select_Indices(ln_difference,average,n):
    selected_indices = [] # Create an empty list to store the left and right index of the blur circle into
    for y in range(n): # Execute for the first 10 column and top 10 rows
        ind = np.argpartition(ln_difference[y], -5)[-5:] # Find top 5 highest ln-differences per row
        ind2 = np.argpartition(ln_difference[:, y], -5)[-5:] # Find top 5 highest ln-differences per column
        for i in range(5): # Go through the top 5 highest ln-differences
            if np.abs(ln_difference[y][ind][i] - average) < 0.2: # Only execute if close to the calculated average
                tuple = (y, ind[i]) # Create a tuple using the indices
                selected_indices.append(tuple) # Store the tuple in the list
            if np.abs(ln_difference[:, y][ind2][i] - average) < 0.2: # Only execute if close to the calculated average
                tuple = (ind2[i], y) # Create a tuple using the indices
                selected_indices.append(tuple) # Store the tuple in the list
    return selected_indices

# Estimate the Depth
def Calculate_Depth(ln_difference,selected_indices):
    depth_sum=0 # Initialize counter variable at 0
    for i in range(len(selected_indices)):
        u_array=selected_indices[i][0] # Grab index from list
        v_array=selected_indices[i][1] # Grab index from list
        u=np.abs(u_array-82)/164 # Determine u frequency
        v=np.abs(v_array-91)/182 # Determine v frequency
        denominator = 2*np.pi**2*((u**2)+(v**2)) # Calculate denominator
        sigma1 = np.sqrt((ln_difference[u_array][v_array]) / (denominator)) # Calculate radius of blur circle
        depth = F * v0 / (v0 - F - sigma1 * f1) # Calculate Depth
        depth_sum+=depth # Add 1 to counter variable
    Average_Depth = depth_sum/len(selected_indices) # Determine final depth estimation
    return(Average_Depth)


