from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from skimage import color # For the convertion from RGB to HSV
import numpy as np


# load the image and convert it to a floating point data type
image = img_as_float(io.imread('img1.PNG'))
image_hsv = color.rgb2hsv(image)
# apply SLIC and extract (approximately) the supplied number of segments
numSegments = 1000
sigma_slic = 2
segments = slic(image, n_segments = numSegments, sigma = sigma_slic)     

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")  
# show the plot
fig.set_size_inches(12,9)
fig.savefig('img1_seg.png',bbox_inches ='tight')
plt.show()


# Number of bin
b = 10
# To store the 100 H_S bin values per segment
H_S_val = np.zeros([np.max(segments)+1,b*b], dtype = float)
# To store the 10 V bin values per segment
# Plus one to account for the zero
V_val = np.zeros([np.max(segments)+1,b], dtype = float)
# Loop over each of the segments

for (i, segVal) in enumerate(np.unique(segments)):
    # Mask only the required locaitons from the image
    spat_cord_seg = np.array(np.where(segments == segVal))
    spat_cord_seg = image_hsv[spat_cord_seg[0,:],spat_cord_seg[1,:]]
    # Retrive the HSV values for each pixel postition in the segment
    h = spat_cord_seg[:,0]
    s = spat_cord_seg[:,1]
    v = spat_cord_seg[:,2]
    # Create a 2D histogram with num of bins b = 10, for HUE and SATURATION
    H_S_hist, xedges, yedges = np.histogram2d(h, s, bins=b, range = [[0,1],[0,1]], normed = True)
    # OneD histogram with num of bins b = 10
    V_hist, xedges= np.histogram(v, bins =b, range =(0,1))
    # Store the feature vectors    
    H_S_val[segVal,:] = np.resize(H_S_hist,(b*b,1)).T
    H_S_val[segVal,:] = H_S_val[segVal,:]/sum(H_S_val[segVal,:])       
    V_val[segVal,:] = V_hist/sum(V_hist)
            
         
         