'''
===============================================================================
Interactive Manga Colorization.
USAGE:
    python main.py <filename>
README FIRST:
    Two windows will show up, one for input and one for output.
    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.
Key '0' - To do intensity continuous colorization
Key '1' - To do pattern continuous colorization
Key '2' - To do stroke preserved colorization
Key '3' - To do Pattern to shading colorization
Key ctrl + 'c' - To stop level set method
===============================================================================
'''
from scipy.misc import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import webcolors
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
from skimage.draw import polygon, polygon_perimeter
from skimage import measure
import cv2
import level_set
import numpy as np
import sys

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
current_former_x,current_former_y = -1,-1
ix, iy = -1, -1
image = 0
r = 0
g = 0
b = 0

def nothing(x):
    pass

# mouse callback function
def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode, r, g, b
 
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
 
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(image,(current_former_x,current_former_y),(former_x,former_y),(b,g,r),5)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(image,(current_former_x,current_former_y),(former_x,former_y),(b,g,r),5)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y

def run():
    global current_former_x,current_former_y,drawing, mode, r, g, b, image
 
	# iter_inner, iter_outer, lamda, alpha, epsilon, sigma, dt, potential_function
	# potential_function="single-well"
    if len(sys.argv) == 2:
        filename = sys.argv[1] 
    image = cv2.imread(filename,True)
    output = image.copy()
	# plt.imshow(image)
	# plt.show()
	# print(image.shape)
    color = np.zeros((10,10,3), np.uint8)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('trackbar', cv2.WINDOW_NORMAL)
	#cv2.namedWindow('trackbar')
	#cv2.resizeWindow('trackbar', (10,10))
    cv2.setMouseCallback('image', paint_draw)
    cv2.createTrackbar('R','trackbar',0,255, nothing)
    cv2.createTrackbar('G','trackbar',0,255, nothing)
    cv2.createTrackbar('B','trackbar',0,255, nothing)
    # while(1):
    #     cv2.imshow('image',image)
    #     if cv2.waitKey(20) & 0xFF == 27:
    #         break
    #     r = cv2.getTrackbarPos('R','image')
    #     g = cv2.getTrackbarPos('G','image')
    #     b = cv2.getTrackbarPos('B','image')
    # cv2.destroyAllWindows()

	# #image1 = np.array(image1,dtype='float32')
    # LS = levelSet(4,50,2,-9,2.0,0.8)
    # print(current_former_x, current_former_y)
    # boundary, F= LS.gradientDescent(image1[:,:,0],current_former_y,current_former_x)
	#print(boundary)
	# img_yuv = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
	# yuv = RGB2YUV(np.asarray([r,g,b]).reshape((3,1)))
	# print(yuv, F.shape)
	# #F= LS.calculateF(img_yuv)
	# xyz = cv2.filter2D(np.square(1-F),-1,yuv[0])
	# img_yuv[:,:,0] = xyz
	# cv2.imshow("sdjkfnskj",cv2.cvtColor(img_yuv, cv2.COLOR_LUV2RGB))
	# cv2.waitKey(0)
    # LS.fillColor(image1,boundary,(b,g,r))
    while(1):
        cv2.imshow('output', output)
        cv2.imshow('image',image)
        cv2.imshow('trackbar', color)
        r = cv2.getTrackbarPos('R','trackbar')
        g = cv2.getTrackbarPos('G','trackbar')
        b = cv2.getTrackbarPos('B','trackbar')
        color[:] = b, g, r
        k = cv2.waitKey(1)

        # key bindings
        if k == 27:         # esc to exit
            break
        elif k == ord('0'): #
            LS = level_set.levelSet(4,10000,2,-9,2.0,0.8)
            curcolor = (b,g,r)
            boundary, F= LS.gradientDescent(output[:,:,0],current_former_y,current_former_x)
            output = LS.fillColor(output,boundary,(curcolor))
        elif k == ord('1'): 
            pass
        elif k == ord('2'): 
            pass
        elif k == ord('3'): 
            pass

    print('Done')

if __name__ == '__main__':
    print(__doc__)
    run()
    cv2.destroyAllWindows()