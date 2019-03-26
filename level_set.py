from scipy.misc import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import webcolors
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
from skimage.draw import polygon, polygon_perimeter
from skimage import measure
import cv2
# import drlse_algo as drlse
import numpy as np

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

class drlse(object):

	def __init__(self, F, lamda, mu, alpha, epsilon, dt, iterations, potential_function, M1, M2, F1):
		self.F = F
		self.lamda = lamda
		self.alpha = alpha
		self.epsilon = epsilon
		self.dt = dt
		self.mu = mu
		self.iter = iterations
		self.potential_function = potential_function
		self.M1 = M1
		self.M2 = M2
		self.F1 = F1

	def sigmoid(self, x):
		return np.exp(x)/(1+ np.exp(x))

	def drlse_edge(self,phi):
		[vy, vx] = np.gradient(self.F)
		# [vy, vx, vz] = np.gradient(self.F)
		for k in range(self.iter):
		    phi = self.applyNeumann(phi)
		    [phi_y, phi_x] = np.gradient(phi)
		    # [phi_y, phi_x, phi_z] = np.gradient(phi)
		    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
		    smallNumber = 1e-10
		    Nx = phi_x / (s + smallNumber)
		    Ny = phi_y / (s + smallNumber)
		    # Nz = phi_z / (s + smallNumber)
		    curvature = self.div(Nx, Ny)
		    if self.potential_function == 'single-well':
		        distRegTerm = filters.laplace(phi, mode='wrap') - curvature
		    elif self.potential_function == 'double-well':
		        distRegTerm = self.distReg_p2(phi)
		    else:
		        print('Error: Wrong choice of potential function. Please input the string "single-well" or "double-well" in the drlse_edge function.')
		    diracPhi = self.Dirac(phi)
		    areaTerm = diracPhi * self.F
		    edgeTerm = diracPhi * (vx * Nx + vy * Ny) + diracPhi * self.F * curvature
		    x = (self.F1 - self.M2)/(self.M1 - self.M2)
		    leakproofterm = self.F*areaTerm*self.sigmoid(x)
		    y =  self.dt * (self.mu * distRegTerm + self.lamda * edgeTerm + self.alpha * areaTerm - leakproofterm*self.alpha)
		    print(np.unique(y))
		    phi = phi + self.dt * (self.mu * distRegTerm + self.lamda * edgeTerm + self.alpha * areaTerm - leakproofterm*self.alpha)

		return phi

	def distReg_p2(self,phi):
	    [phi_y, phi_x] = np.gradient(phi)
	    # [phi_y, phi_x, phi_z] = np.gradient(phi)
	    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
	    a = (s >= 0) & (s <= 1)
	    b = (s > 1)
	    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
	    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))
	    return self.div(dps * phi_x - phi_x, dps * phi_y - phi_y) + filters.laplace(phi, mode='wrap')

	def div(self,nx, ny):
	    [junk, nxx] = np.gradient(nx)
	    [nyy, junk] = np.gradient(ny)
	    # [junk, nxx, nzz] = np.gradient(nx)
	    # [nyy, junk, nzz] = np.gradient(ny)
	    return nxx + nyy

	def Dirac(self,x):
	    f = (1 / 2 / self.epsilon) * (1 + np.cos(np.pi * x / self.epsilon))
	    b = (x <= self.epsilon) & (x >= -self.epsilon)
	    return f * b

	def applyNeumann(self,f):
	    [ny, nx] = f.shape
	    # [ny, nx, nz] = f.shape
	    g = f.copy()
	    g[0, 0] = g[2, 2]
	    g[0, nx-1] = g[2, nx-3]
	    g[ny-1, 0] = g[ny-3, 2]
	    g[ny-1, nx-1] = g[ny-3, nx-3]

	    g[0, 1:-1] = g[2, 1:-1]
	    g[ny-1, 1:-1] = g[ny-3, 1:-1]

	    g[1:-1, 0] = g[1:-1, 2]
	    g[1:-1, nx-1] = g[1:-1, nx-3]
	    return g



class levelSet(object):

	def __init__(self, drlse_iter, gradient_iter, lamda, alpha, epsilon, sigma, dt=1, potential_function="double-well"):
		self.lamda = lamda
		self.alpha = alpha
		self.epsilon = epsilon
		self.sigma = sigma
		self.dt = dt
		self.mu = 0.2/self.dt
		self.drlse_iter = drlse_iter
		self.gradient_iter = gradient_iter
		self.potential_function = potential_function

	def initializePhiAtScribble(self,image,x,y):
		c0 = 4
		phi = c0 * np.ones(image.shape)
		phi[x-5:x+5, y-3:y+3] = -c0
		return phi

	def visualization(self,image,phi):
		contours = measure.find_contours(phi, 0)
		fig2 = plt.figure(2)
		fig2.clf()
		ax2 = fig2.add_subplot(111)
		ax2.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
		for n, contour in enumerate(contours):
			ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)
			#print(contour)
		return contour

	def calculateF(self,image):
		img_smooth = filters.gaussian_filter(image, self.sigma)
		[Iy, Ix] = np.gradient(img_smooth)
		f = np.square(Ix) + np.square(Iy)
		f1 = np.sqrt(f)
		#print(np.unique(f))
		return 1 / (1+f), np.max(f1), np.min(f1), f1

	def fillColor(self,image,boundary,rgb):
		# boundary[:,[0,1]] = boundary[:,[1,0]]
		#print(boundary)
		img = image.copy()
		rr, cc = polygon(boundary[:,0], boundary[:,1], image.shape)
		image[rr,cc,:] = rgb
		rr, cc = polygon_perimeter(boundary[:,0], boundary[:,1], image.shape)
		image[rr,cc,:] = rgb
		# rgb = (webcolors.name_to_rgb(col)[0],webcolors.name_to_rgb(col)[1],webcolors.name_to_rgb(col)[2])
		# print(rgb)
		# for x in range(image.shape[0]):
		# 	for y in range(image.shape[1]):
		# 		if(cv2.pointPolygonTest(boundary,(x,y),True)>=0):
		# 			image[y][x][0] = rgb[0]
		# 			image[y][x][1] = rgb[1]
		# 			image[y][x][2] = rgb[2]
		#cv2.fillPoly(image, pts =[boundary], color=rgb)
		#cv2.imwrite("123.png",image)
		#cv2.imshow("sdjkfnskj",image+img)
		#cv2.waitKey(0)

	def gradientDescent(self,image,x,y):
		phi = self.initializePhiAtScribble(image,x,y)
		F, M1, M2, F1 = self.calculateF(image)
		lse = drlse(F, self.lamda, self.mu, self.alpha, self.epsilon, self.dt, self.drlse_iter, self.potential_function, M1, M2, F1)
		for n in range(self.gradient_iter):
			phi = lse.drlse_edge(phi)
			if np.mod(n, 2) == 0:			
				boundary = self.visualization(image.copy(),phi)
				plt.pause(0.3)
		return np.int32(boundary), F
		# plt.pause(5)
def RGB2YUV( rgb ):
     
    m = np.array([[ 0.29900, 0.587,  0.114],
                 [-0.14713, -0.28886, 0.436],
                 [ 0.615, -0.51499, -0.10001]])
     
    yuv = np.dot(m,rgb)
    print(yuv.shape, rgb.shape)
    return yuv


def main():
	global current_former_x,current_former_y,drawing, mode, r, g, b, image
 
	# iter_inner, iter_outer, lamda, alpha, epsilon, sigma, dt, potential_function
	# potential_function="single-well"

	image = cv2.imread('13.png',True)
	image1 = image.copy()
	# plt.imshow(image)
	# plt.show()
	# print(image.shape)
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	#cv2.namedWindow('trackbar')
	#cv2.resizeWindow('trackbar', (10,10))
	cv2.setMouseCallback('image', paint_draw)
	cv2.createTrackbar('R','image',0,255, nothing)
	cv2.createTrackbar('G','image',0,255, nothing)
	cv2.createTrackbar('B','image',0,255, nothing)
	while(1):
		cv2.imshow('image',image)
		if cv2.waitKey(20) & 0xFF == 27:
			break
		r = cv2.getTrackbarPos('R','image')
		g = cv2.getTrackbarPos('G','image')
		b = cv2.getTrackbarPos('B','image')
	cv2.destroyAllWindows()

	#image1 = np.array(image1,dtype='float32')
	LS = levelSet(4,50,2,-9,2.0,0.8)
	print(current_former_x, current_former_y)
	boundary, F= LS.gradientDescent(image1[:,:,0],current_former_y,current_former_x)
	#print(boundary)
	img_yuv = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
	yuv = RGB2YUV(np.asarray([r,g,b]).reshape((3,1)))
	print(yuv, F.shape)
	#F= LS.calculateF(img_yuv)
	xyz = cv2.filter2D(np.square(1-F),-1,yuv[0])
	img_yuv[:,:,0] = xyz
	cv2.imshow("sdjkfnskj",cv2.cvtColor(img_yuv, cv2.COLOR_LUV2RGB))
	cv2.waitKey(0)
	LS.fillColor(image1,boundary,(b,g,r))


if __name__ == '__main__':
	main()
