import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

def read_image(img_path):
    # Read the image from the file
    image = cv2.imread(img_path)
    ''' 
    Since OpenCV reads images in BGR format and Matplotlib displays
    images in RGB format, we need to convert the image
    '''
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def display_images_and_histograms(images, titles):
    """
    Displays multiple images and their histograms in a single window using Matplotlib.
    :param images: List of images to display
    :param titles: List of titles corresponding to the images
    """
    num_images = len(images)
    plt.figure(figsize=(15, 10))
    
    for i in range(num_images):
        # Display the image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        plt.title(titles[i])
        plt.axis('off')  # Hide the axis

        # Display the histogram
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.hist(images[i].ravel(), 256, [0, 256])
        plt.title(titles[i] + ' Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def apply_gaussian_blur(image, kernel_size):
    '''
    Applies Gaussian blur to an image using the specified kernel size.
    :param image: Input image
    :param kernel_size: Size of the Gaussian kernel
    :return: Blurred image
    '''
    # Create a Gaussian kernel using OpenCV
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma=0)
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T  # To make it 2D

    # Apply the kernel to the image
    blurred_image = cv2.filter2D(image, -1, gaussian_kernel)

    return blurred_image

def apply_edge_detection(image):
    '''
    Applies edge detection using the Sobel operator 
    :param image: Input image
    :return: Image with edges detected 
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Sobel operator to detect edges 
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    return sobel_combined

def apply_sharpening(image):
    '''
    Applies a sharpening filter to an image.
    :param image: Input image
    :return: Sharpened image
    '''
    # Define a sharpening kernel 
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    
    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

    return sharpened_image

def apply_histogram_equalization(image):
    '''
    Applies histogram equalization to an image.
    :param image: Input image
    :return: Equalized image
    '''
    # Convert the image to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # Apply histogram equalization
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

    # Convert the image back to RGB color space
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

    return equalized_image

def menu():
    print("Image Processing Menu:")
    print("1. Apply Gaussian Blur")
    print("2. Apply Edge Detection")
    print("3. Apply Image Sharpening")
    print("4. Apply Histogram Equalization")
    print("5. Apply All Processing Techniques")
    print("6. Exit")
    choice = input("Enter your choice (1-6): ")
    return choice

def main():
    parser = argparse.ArgumentParser(description="Image Processing with OpenCV and NumPy")
    parser.add_argument("image", help="Path to the image file")
    args = parser.parse_args()

    image_path = args.image
    image_rgb = read_image(image_path)

    while True:
        choice = menu()
        
        if choice == '1':
            kernel_size = int(input("Enter the kernel size for Gaussian Blur (odd number): "))
            blurred_image = apply_gaussian_blur(image_rgb, kernel_size)
            display_images_and_histograms([image_rgb, blurred_image], ['Original Image', 'Gaussian Blurred Image'])
        
        elif choice == '2':
            edges = apply_edge_detection(image_rgb)
            display_images_and_histograms([image_rgb, edges], ['Original Image', 'Edge Detection'])
        
        elif choice == '3':
            sharpened_image = apply_sharpening(image_rgb)
            display_images_and_histograms([image_rgb, sharpened_image], ['Original Image', 'Sharpened Image'])
        
        elif choice == '4':
            equalized_image = apply_histogram_equalization(image_rgb)
            display_images_and_histograms([image_rgb, equalized_image], ['Original Image', 'Histogram Equalized Image'])
        
        elif choice == '5':
            kernel_size = int(input("Enter the kernel size for Gaussian Blur (odd number): "))
            blurred_image = apply_gaussian_blur(image_rgb, kernel_size)
            edges = apply_edge_detection(image_rgb)
            sharpened_image = apply_sharpening(image_rgb)
            equalized_image = apply_histogram_equalization(image_rgb)
            display_images_and_histograms([image_rgb, blurred_image, edges, sharpened_image, equalized_image], 
                           ['Original Image', 'Gaussian Blurred Image', 'Edge Detection', 'Sharpened Image', 'Histogram Equalized Image'])
        
        elif choice == '6':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
