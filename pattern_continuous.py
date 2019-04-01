import os
import cv2
from scipy.misc import imread
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
from skimage import measure
import matplotlib.pyplot as plt
import copy
import numpy as np
from skimage.draw import polygon, polygon_perimeter

from skimage.filters import gabor_kernel
from scipy import ndimage as nd

kernels = []
for ksize in [15,21,31,63]:
    for theta in np.arange(0, np.pi, np.pi / 6):
        kern = cv2.getGaborKernel((ksize, ksize), 4., theta, 12.0, 0.5, 0, ktype=cv2.CV_32F) #13 was 10(4th parameter)
        # increase to consider narrow regions
        kern /= 1.5*kern.sum()
        kernels.append(kern)
# print(len(kernels))

def process(img, filters):
    accum = []
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
#         np.maximum(accum, fimg, accum)
        accum.append(fimg)
    return accum



import numpy as np
FF = np.zeros((1,1),dtype=np.double)



class drlse_pattern(object):

    def __init__(self, F, lamda, mu, alpha, epsilon, dt, iterations, potential_function):
        self.F = F
        self.lamda = lamda
        self.alpha = alpha
        self.epsilon = epsilon
        self.dt = dt
        self.mu = mu
        self.iter = iterations
        self.potential_function = potential_function

    def drlse_edge(self,phi):
        [vy, vx] = np.gradient(self.F)
        for k in range(self.iter):
            phi = self.applyNeumann(phi)
            [phi_y, phi_x] = np.gradient(phi)
            s = np.sqrt(np.square(phi_x) + np.square(phi_y))
            smallNumber = 1e-10
            Nx = phi_x / (s + smallNumber)
            Ny = phi_y / (s + smallNumber)
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
            phi = phi + self.dt * (self.mu * distRegTerm + self.lamda * edgeTerm + self.alpha * areaTerm)
        return phi

    def distReg_p2(self,phi):
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        a = (s >= 0) & (s <= 1)
        b = (s > 1)
        ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
        dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))
        return self.div(dps * phi_x - phi_x, dps * phi_y - phi_y) + filters.laplace(phi, mode='wrap')

    def div(self,nx, ny):
        [junk, nxx] = np.gradient(nx)
        [nyy, junk] = np.gradient(ny)
        return nxx + nyy

    def Dirac(self,x):
        f = (1 / 2 / self.epsilon) * (1 + np.cos(np.pi * x / self.epsilon))
        b = (x <= self.epsilon) & (x >= -self.epsilon)
        return f * b

    def applyNeumann(self,f):
        [ny, nx] = f.shape
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



class levelSet_pattern(object):

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
        c0 = 3
        phi = c0 * np.ones(image.shape)
        w = 8
        phi[x-w:x+w, y-w:y+w] = -c0
        return phi

    def visualization(self,image,phi):
        fig2 = plt.figure(2)
        fig2.clf()
        contours = measure.find_contours(phi, 0)
        ax2 = fig2.add_subplot(111)
        ax2.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
        for n, contour in enumerate(contours):
            ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)

    def pattern2shading(self, image, rgb):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color = np.array(rgb)
        colored_img = np.ones((self.phi.shape[0],self.phi.shape[1],3), dtype=np.uint8)
        colored_img[self.phi<0] = color
        # plt.imshow(colored_img)
        # plt.pause(0.2)

        yuv_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2YUV)
        # plt.imshow( LS.FI[60:100] )
        # plt.pause(0.2)
        plt.imshow(( (1-self.FI)**2 ))
        plt.pause(0.2)
        # print(np.unique(self.FI,return_counts=True))
        y=0
        # y=0
        s = cv2.boxFilter(yuv_img[:,:,y],-1,(8,8))
        s = s/np.max(s)
        # plt.imshow(s)
        # plt.pause(0.1)
        y_image = yuv_img[:,:,y]
        y_image[self.phi!=0] = y_image[self.phi!=0] * s[self.phi!=0]

        # cv2.imshow('coloured',colored_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        colored_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
        colored_img[:,:,0] *= np.array(img/10,dtype=np.uint8)
        colored_img[:,:,1] *= np.array(img/10,dtype=np.uint8)
        colored_img[:,:,2] *= np.array(img/10,dtype=np.uint8)
        # plt.imshow(colored_img)
        # plt.pause(0.2)
        res = np.ones((self.phi.shape[0],self.phi.shape[1],3), dtype=np.uint8)
        res[self.phi>0] = colored_img[self.phi>0] + image[self.phi>0]
        res[self.phi<0] = colored_img[self.phi<0]
        cv2.imwrite("res_shading_4.png", res)
        # plt.imshow(colored_img)
        # plt.pause(0.2)
        return res

    def strokepreserving(self, image, rgb):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color = np.array(rgb)
        colored_img = np.ones((self.phi.shape[0],self.phi.shape[1],3), dtype=np.uint8)
        colored_img[self.phi<0] = color
        # plt.imshow(colored_img)
        # plt.pause(0.2)

        yuv_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2YUV)
        # plt.imshow( LS.FI[60:100] )
        # plt.pause(0.2)
        plt.imshow(( (1-self.FI)**2 ))
        plt.pause(0.2)
        # print(np.unique(self.FI,return_counts=True))
        y=0
        yuv_img[:,:,y] = yuv_img[:,:,y] * ( (1-self.FI)**2 )

        colored_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
        # cv2.imshow('coloured',colored_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        colored_img[:,:,0] *= np.array(img/10,dtype=np.uint8)
        colored_img[:,:,1] *= np.array(img/10,dtype=np.uint8)
        colored_img[:,:,2] *= np.array(img/10,dtype=np.uint8)
        plt.imshow(colored_img)
        plt.pause(0.2)
        res = np.ones((self.phi.shape[0],self.phi.shape[1],3), dtype=np.uint8)
        res[self.phi>0] = colored_img[self.phi>0] + image[self.phi>0]
        res[self.phi<0] = colored_img[self.phi<0]
        cv2.imwrite("res_shading_4.png", res)
        # plt.imshow(colored_img)
        # plt.pause(0.2)
        return res

    def fillColor(self,img,phi,rgb):
        image = img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        mask = np.zeros(image.shape, dtype=np.bool)
        mask[phi<0] = True

        color = rgb
        colored_img = np.ones((self.phi.shape[0],self.phi.shape[1],3), dtype=np.uint8)
        colored_img[~mask] = image[~mask]/3,image[~mask]/3,image[~mask]/3
        colored_img[phi<0] = rgb

        yuv_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2YUV)

        y=0
        yuv_img[:,:,y] = yuv_img[:,:,y] * ( (1-self.FI)**2 )

        colored_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

        colored_img[:,:,0] *= np.array(image/10,dtype=np.uint8)
        colored_img[:,:,1] *= np.array(image/10,dtype=np.uint8)
        colored_img[:,:,2] *= np.array(image/10,dtype=np.uint8)

        return coloured_img


    def calculateF_edge(self,image):
#         [Iy, Ix] = np.gradient(image)
        img_smooth = filters.gaussian_filter(image, self.sigma)
        [Iy, Ix] = np.gradient(img_smooth)
        F = np.square(Ix) + np.square(Iy)
#         print((1/(1+F)).shape)
        return 1 / (1+F)

    def compute_feats(self, filtered):
        feats = np.zeros((len(filtered), 2), dtype=np.double)
        for k, fltr in enumerate(filtered):
            feats[k, 0] = fltr.mean()
            feats[k, 1] = fltr.var()
        return feats

    def calculateF_pattern(self,image,x,y):
        w = 8
        filtered = process(image, kernels)
        window_scribble = [f[x-w:x+w, y-w:y+w] for f in filtered]
        scribble_feats = self.compute_feats(window_scribble)

        top, bottom, left, right = w, w-1, w, w-1
        WHITE = [255, 255, 255]
        padded = [cv2.copyMakeBorder(f, top , bottom, left, right, cv2.BORDER_REPLICATE, value=WHITE) \
                  for f in filtered]

        global FF
        if FF.shape[0] == 1:
            print("computing pattern features")
            F = np.zeros(image.shape, dtype=np.double)
            step = 1
            for i in range(w, padded[0].shape[0]-w, step):
                for j in range(w, padded[0].shape[1]-w, step):

                    feats = self.compute_feats([f[i-w:i+w, j-w:j+w] for f in padded])
    #                 print(i,j)
                    F[i-w:i-w+step, j-w:j-w+step] = np.sum((feats - scribble_feats)**2)
#                     F[i-w:i-w+step, j-w:j-w+step] = np.sqrt(F[i-w:i-w+step, j-w:j-w+step])
            FF = F
            print("done!")
        else:
            F = copy.deepcopy(FF)

#         F = (F - np.mean(F)) / np.std(F)
#         print(F)
#         F[F<17000] = 1
#         F[F>=17000] = 1000000
#         print(np.unique(F,return_counts=True))
#         print("")
#         X = 1./(1+F)
#         X = (X - np.mean(X)) / np.std(X)
#         print(X)
        # plt.imshow(F)
        # plt.pause(0.3)
        # plt.imshow(1 / (1+F))
        # plt.pause(0.3)
        return 1 / (1+F)

    def gradientDescent(self,image,x,y):
        self.phi = self.initializePhiAtScribble(image,x,y)
        self.FI = self.calculateF_edge(image)
        F = self.calculateF_pattern(image,x,y)
        lse = drlse_pattern(F, self.lamda, self.mu, self.alpha, self.epsilon, self.dt, self.drlse_iter, self.potential_function)
        try:
            for n in range(self.gradient_iter):
                self.phi = lse.drlse_edge(self.phi)
    #             [phi_y, phi_x] = np.gradient(phi)
    #             dphi = np.sqrt(np.square(phi_x) + np.square(phi_y))
    #             phi = phi + self.dt * F * dphi
                if n % 10 == 0:
                    boundary = self.visualization(image,self.phi)
                    plt.pause(1)
        except KeyboardInterrupt:
            pass

        return self.phi, F
