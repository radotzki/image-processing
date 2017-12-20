# IMPR 2017, IDC
# ex3 driver

import numpy as np
import cv2
import matplotlib.pyplot as plt

import ex3

# to get fixed set of random numbers
np.random.seed(seed=0)

   
def test_1(imageName):
    
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
    imageEdges = cv2.Canny (img,100,200)
    
    votesThresh = 60
    distThresh = 15
    radius = range(5,31,5)
    circles = ex3.HoughCircles(imageEdges,radius, votesThresh, distThresh)
          
    f, (ax1, ax2)  = plt.subplots(1, 2, sharex='col')
    ax1.imshow(img, cmap='gray',vmin=0, vmax=255), ax1.set_title('Original+ detected circles')
    ax2.imshow(imageEdges, cmap='gray',vmin=0, vmax=255), ax2.set_title('Canny edges')
    
    circle = []
    for y,x,r,val in circles:
        circle.append(plt.Circle((x,y),r,color=(1,0,0),fill=False))
        ax1.add_artist(circle[-1])
    plt.show()


def test_2(imageName, noiseStd=0.1):
    
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
    
    # add noiseStd percent Gaussian noise 
    imgNoisy = img + np.random.normal(0, noiseStd*255, size=img.shape)
    imgNoisy [imgNoisy<0] = 0
    imgNoisy [imgNoisy>255] = 255
    imgNoisy= np.uint8(np.round(imgNoisy))
    
    spatial_std = 1
    range_std = noiseStd*255
    
      
    imgClean = ex3.bilateralFilter (imgNoisy, spatial_std, range_std)
    f, (ax1, ax2, ax3, ax4)  = plt.subplots(1, 4, sharex='col')
    ax1.imshow(img, cmap='gray',vmin=0, vmax=255), ax1.set_title('Original')
    ax2.imshow(imgNoisy, cmap='gray',vmin=0, vmax=255), ax2.set_title('Noisy')
    ax3.imshow(imgClean, cmap='gray',vmin=0, vmax=255), ax3.set_title('After bilateral filter')
    ax4.imshow(imgNoisy-np.float32(imgClean), cmap='gray',vmin=-50, vmax=50), ax4.set_title('Removed noise')
    
    # with large rangeStd -> effectivly Gaussian smoothing
    range_std = 100*noiseStd*255
    imgClean = ex3.bilateralFilter (imgNoisy, spatial_std, range_std)
    f, (ax1, ax2, ax3, ax4)  = plt.subplots(1, 4, sharex='col')
    ax1.imshow(img, cmap='gray',vmin=0, vmax=255), ax1.set_title('Original')
    ax2.imshow(imgNoisy, cmap='gray',vmin=0, vmax=255), ax2.set_title('Noisy')
    ax3.imshow(imgClean, cmap='gray',vmin=0, vmax=255), ax3.set_title('After \'Guassian\' filter')
    ax4.imshow(imgNoisy-np.float32(imgClean), cmap='gray',vmin=-50, vmax=50), ax4.set_title('Removed noise')
    
   
    plt.show()


if __name__ == "__main__":
    
    # test 1.
    imageName = './Images/coins.tif'
    test_1(imageName)

    imageName = './Images/cameraman.tif'
    test_2(imageName,0.05)