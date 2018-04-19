# coding=gbk
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    leftLines, rightLines = [], []
    maxXright,maxXleft = 0,0
    maxYright,maxYleft = 0,0
    averageX1right,averageX2right = [],[]
    averageY1right,averageY2right = [],[]

    averageX1left,averageX2left = [],[]
    averageY1left,averageY2left = [],[]

    for line in lines:
        for x1,y1,x2,y2 in line:
             if (x2 - x1) is not 0:
                k = (y2-y1)/(x2-x1)
                if k>0 :
                    averageX1right.append(x1)
                    averageX2right.append(x2)
                    averageY1right.append(y1)
                    averageY2right.append(y2)

                else :
                    averageX1left.append(x1)
                    averageX2left.append(x2)
                    averageY1left.append(y1)
                    averageY2left.append(y2)

    averageX1right = averageX1right+averageX2right
    averageY1right = averageY1right+averageY2right

    positionXmax = averageX1right.index(max(averageX1right))
    positionYmax = averageY1right.index(max(averageY1right))
    positionXmin = averageX1right.index(min(averageX1right))
    positionYmin = averageY1right.index(min(averageY1right))

    cv2.line(img,(averageX1right[positionXmax], averageY1right[positionXmax]), (averageX1right[positionXmin], averageY1right[positionXmin]), color, thickness=8)

    averageX1left = averageX1left+averageX2left
    averageY1left = averageY1left+averageY2left

    positionXmax = averageX1left.index(max(averageX1left))
    positionYmax = averageY1left.index(max(averageY1left))
    positionXmin = averageX1left.index(min(averageX1left))
    positionYmin = averageY1left.index(min(averageY1left))

    cv2.line(img,(averageX1left[positionXmax], averageY1left[positionXmax]), (averageX1left[positionXmin], averageY1left[positionXmin]), color, thickness=8)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


import os
os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
def imageSaveOutput(image,name,number):
    """
    funciton used to save image in 'test_images_output' with different names
    """
    FileName = name +" "+number
    mpimg.imsave("test_images_output"+'//'+FileName,image)
    return 0;

#mkdir to save output images
originalFileName = os.listdir("test_images/")
cur_path = os.path.abspath(os.curdir)
save_path = cur_path+'\\'+"test_images_output"
if not os.path.exists(save_path):
    os.mkdir(save_path)


#handle image one by one
for i in originalFileName :
    image = mpimg.imread("test_images/"+i)
    grayImage = grayscale(image)
    blurImage = gaussian_blur(grayImage,5)
    cannyImage = canny(blurImage,50,150)

    #mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 325),(520,325),(imshape[1],imshape[0])]], dtype=np.int32)
    interestImage = region_of_interest(cannyImage,vertices)


    #houghLines
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments

    lineImage = hough_lines(interestImage, rho, theta, threshold, min_line_len, max_line_gap)

    #add img
    outputImage = weighted_img(lineImage,image)

    #save iamge
    imageSaveOutput(grayImage,"gray",i)
    imageSaveOutput(blurImage,"blur",i)
    imageSaveOutput(cannyImage,"canny",i)
    imageSaveOutput(interestImage,"interest",i)
    imageSaveOutput(lineImage,"houghLine",i)
    imageSaveOutput(outputImage,"output",i)
    #plt.imshow(blurImage)

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    grayImage = grayscale(image)
    blurImage = gaussian_blur(grayImage,5)
    cannyImage = canny(blurImage,50,150)

    #mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 325),(520,325),(imshape[1],imshape[0])]], dtype=np.int32)
    interestImage = region_of_interest(cannyImage,vertices)


    #houghLines
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments

    lineImage = hough_lines(interestImage, rho, theta, threshold, min_line_len, max_line_gap)

    #add img
    outputImage = weighted_img(lineImage,image)
    return outputImage

#mkdir to save output images
cur_path = os.path.abspath(os.curdir)
save_path = cur_path+'\\'+"test_videos_output"
if not os.path.exists(save_path):
    os.mkdir(save_path)

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
