# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from skimage import color

image = img_as_float(io.imread("Dev_Img_1.png"))
image_lab = color.rgb2hsv(image)

# loop over the number of segments
numSegments = 1000
# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(image, n_segments = numSegments, sigma = 2)

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")
 
# show the plots
fig.set_size_inches(12,9)
fig.savefig('output1.png',bbox_inches='tight')
plt.show()