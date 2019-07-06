import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import math


def main():

    # initialization of constants
    beta = 0.5
    c = 0.08

    # load and prepare image
    image_rgb = cv2.imread('../data/coronaries.jpg')
    image = convert2gray(image_rgb)

    scales = [1.0, 1.5, 2.0, 3.0]
    images_vesselness = []
    for s in scales:

        images_vesselness.append(calculate_vesselness_2d(image, s, beta, c))

    result = compute_scale_maximum(images_vesselness)
    show_four_scales(image, result, images_vesselness, scales)


# calculate the vesselness filter image (Frangi 1998)
def calculate_vesselness_2d(image, scale, beta, c):

    # create empty result image
    vesselness = np.zeros(image.shape)
    print(image.shape)

    # compute the Hessian for each pixel
    H = compute_hessian(image, scale)

    # get the eigenvalues for the Hessians
    eigenvalues = compute_eigenvalues(H)
    lambda1=eigenvalues[:,:,0]
    lambda2=eigenvalues[:,:,1]

    print('Computing vesselness...')

    # compute the vesselness measure for each pixel
    # TODO: loop over the pixels to compute the vesselness image
    # Hint: use the function vesselness_measure (implement it first below)
    width, height = image.shape
    for x in range(width):
        for y in range(height):
            vesselness=vesselness_measure(lambda1,lambda2,beta,c)

    print('...done.')
    return vesselness




def compute_hessian(image, sigma):

    # gauss filter the input with given sigma
    # TODO: filter image using sigma and zero padding (filter mode 'constant')
    
    image_gauss = gaussian_filter(image, sigma, mode='constant',cval=0.0) # replace None by your result

    print('Computing Hessian...')

    # gradient calculation
    # TODO: compute first order gradient
    
    f_ux=convolve2d( image_gauss,np.array([[1,-1]]), mode='same')
    f_x=gaussian_filter(f_ux, sigma, mode='constant',cval=0.0)
    f_uy=convolve2d( image_gauss,np.array([[1],[-1]]), mode='same')
    f_y=gaussian_filter(f_uy, sigma, mode='constant',cval=0.0)

    # Create components of the Hessian Matrix [dx2 dxy][dyx dy2]
    # TODO: compute all partial second derivatives
    f_xx=convolve2d( f_x,np.array([[1,-1]]), mode='same')
    f_yy=convolve2d( f_y,np.array([[1],[-1]]), mode='same')
    f_yx=convolve2d( f_x,np.array([[1],[-1]]), mode='same')
    f_xy=convolve2d( f_y,np.array([[1,-1]]), mode='same')
    

    # scale normalization -> multiply the hessian components with sigma^2
    # TODO: normalize as stated
    f_xx=np.multiply(f_xx,np.square(sigma))
    f_xy=np.multiply(f_xy,np.square(sigma))
    f_yx=np.multiply(f_yx,np.square(sigma))
    f_yy=np.multiply(f_yy,np.square(sigma))
    # save values in a single array
    H = np.empty((np.shape(image_gauss)[0], np.shape(image_gauss)[1], 2, 2))

    # TODO: fill the Hessian with the proper values from above
    H[:,:,0,0]=f_xx
    H[:,:,0,1]=f_xy
    H[:,:,1,0]=f_yx
    H[:,:,1,1]=f_yy
    
    print('...done.')
    return H


# create array for the eigenvalues and compute them
def compute_eigenvalues(hessian):

    evs = np.empty((np.shape(hessian)[0], np.shape(hessian)[1], 2))
    print('Computing eigenvalues, this may take a while...')

    # TODO: implement the computation of the eigenvalues
    # TODO (Hint): make use of np.linalg.eig(...)
    evs,evc=np.linalg.eig(hessian)
    
    
    
    
   

    print('...done.')
    return evs


# calculate the 2-D vesselness measure (see Frangi paper or course slides)
def vesselness_measure(lambda1, lambda2, beta, c):
    
    
    # ensure lambda1 >= lambda2
    lambda1, lambda2 = sort_descending(lambda1, lambda2)
    
    # the vesselness measure is zero if lambda1 is positive (inverted/dark vessel)
    if lambda1>0 :
        v=0#dark vessel
    # if both eigenvalues are zero, set RB and S to zero, otherwise compute them as shown in the course
    elif (lambda1==0 and lambda2==0):
        rb=0
        s=0       
        v=math.exp(-1*np.square(rb)/(2*np.square(beta)))*(1-math.exp(-1*np.square(s)/(2*np.square(c))))
        # TODO: implement the vesselness measure and take care of lambda1 being zero
    else:
        rb=lambda2/lambda1
        s=math.sqrt(lambda1*lambda1+lambda2*lambda2)
        v=math.exp(-1*np.square(rb)/(2*np.square(beta)))*(1-math.exp(-1*np.square(s)/(2*np.square(c))))
# dummy result
        
    return v


# takes a list of vesselness images and returns the pixel-wise maximum as a result
def compute_scale_maximum(image_list):

    result = image_list[0]
    print('Computing maximum...')
    

    # TODO: compute the image that takes the PIXELWISE maximum from all images in image_list
    for i in range(len(image_list)): 
        result=np.maximum(image_list[i],result)
        

    print('...done.')
    return result


# convert to gray scale and normalize for float
# (OpenCV treats color pixels as BGR)
def convert2gray(image_rgb):

    temp = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image_gray = temp.astype(np.float32) / 255.0

    return image_gray


# rearrange pair of values in descending order
def sort_descending(value1, value2):

    if np.abs(value1) < np.abs(value2):
        buf = value2
        value2 = value1
        value1 = buf
    
   

    return value1, value2


# special function to show the images from this exercise
def show_four_scales(original, result, image_list, scales):

    plt.figure('vesselness')

    prepare_subplot_image(original, 'original', 1)
    prepare_subplot_image(image_list[0], 'sigma = '+str(scales[0]), 2)
    prepare_subplot_image(image_list[1], 'sigma = '+str(scales[1]), 3)
    prepare_subplot_image(result, 'result', 4)
    prepare_subplot_image(image_list[2], 'sigma = '+str(scales[2]), 5)
    prepare_subplot_image(image_list[3], 'sigma = '+str(scales[3]), 6)

    plt.show()


# helper function
def prepare_subplot_image(image, title='', idx=1):

    if idx > 6:
        return

    plt.gcf()
    plt.subplot(2, 3, idx)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='gray', vmin=0, vmax=np.max(image))


# function for displaying an image and waiting for user input
def show_image(i, t, destroy_windows=True):

    cv2.imshow(t, i)

    print('Press a key to continue...')
    cv2.waitKey(0)

    if destroy_windows:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
