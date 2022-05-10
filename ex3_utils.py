import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 328601018

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

def NormalizeData(data):
    """
    return the array normalized to numbers between 0 and 1
    :param data:
    :return:
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))



def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """

    ker = np.array([[1, 0, -1]])
    rep = int(np.floor(win_size/2))


    im2pad = cv2.copyMakeBorder(im2, rep, rep, rep, rep, cv2.BORDER_REPLICATE, None, value=0)
    Ix = cv2.filter2D(im2pad, -1, ker, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2pad, -1, ker.T, borderType=cv2.BORDER_REPLICATE)
    # Ix = cv2.Sobel(im2, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
    # Iy = cv2.Sobel(im2, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)

    It=im2-im1

    # Ixpad=cv2.copyMakeBorder(Ix, rep, rep, rep, rep, cv2.BORDER_REPLICATE, None, value=0)
    # Iypad = cv2.copyMakeBorder(Iy, rep, rep, rep, rep, cv2.BORDER_REPLICATE, None, value=0)
    # Itpad = cv2.copyMakeBorder(It, rep, rep, rep, rep, cv2.BORDER_REPLICATE, None, value=0)
    # f, ax = plt.subplots(1, 5)
    # ax[0].imshow(im1)
    # ax[1].imshow(Ix)
    # ax[2].imshow(Iy)
    # ax[3].imshow(im2)
    # ax[4].imshow(It)
    # plt.show()


    z=0
    origpoints = np.array([])
    newpoints = np.array([])
    # start= int(np.floor(win_size/2))
    start=win_size
    for i in range(step_size,im2.shape[0],step_size):
        for j in range(step_size, im2.shape[1],step_size):
            try:
                # small=[]
                # # template match the best win_size*win_size square in the area step_size*step_size
                # for x in range(i-start,i+step_size-start):
                #     for y in range(j-start,j+step_size-start):
                #         if(x-rep>=0 and x+rep+1<im2.shape[0] and y-rep>=0 and y+rep+1<im2.shape[1]):
                #             # sq=((It[x-rep:x+rep+1, y-rep:y+rep+1])).sum()
                #             i1=im1[x-rep:x+rep+1, y-rep:y+rep+1]
                #             i1a=i1.reshape(1,25)
                #             i1b=i1.reshape(25,1)
                #             i2=im2[x-rep:x+rep+1, y-rep:y+rep+1]
                #             i2a = i2.reshape(1, 25)
                #             i2b = i2.reshape(25, 1)
                #             ncc=float(i1a.dot(i2b)/(i1a.dot(i1b))*(i2a.dot(i2b)))
                #             if len(small) == 0:
                #                 small.append((ncc, x, y))
                #             else:
                #                 if small[0][0]<ncc:
                #                     small[0]=(ncc, x, y)
                # print(small)

                # find the u and v for the best matched area
                # A = np.array([])
                # B = np.array([])


                B = -It[i-rep : i+rep+1, j-rep : j+rep+1]
                # B = -It[small[0][1]-rep : small[0][1]+rep+1, small[0][2]-rep :small[0][2]+rep+1]
                B = B.reshape(25, 1)
                A1 = Ix[i-rep : i+rep+1, j-rep : j+rep+1].reshape(25,1)
                A2 = Iy[i-rep : i+rep+1, j-rep : j+rep+1].reshape(25,1)
                # A1 = Ix[small[0][1]-rep : small[0][1]+rep+1, small[0][2]-rep : small[0][2]+rep+1].reshape(25,1)
                # A2 = Iy[small[0][1]-rep : small[0][1]+rep+1, small[0][2]-rep : small[0][2]+rep+1].reshape(25,1)
                A = np.stack((A1,A2))
                A = A.reshape(25,2)
                AT = A.T
                ATA = AT@A
                e, e1 = np.linalg.eig(ATA)
                e = np.sort(e)
                # make sure the eigen values are ok
                if e[1] >= e[0] > 1 and e[1]/e[0] < 100:
                    v = np.linalg.inv(ATA) @ AT @ B
                    # if(v[0] != 0 and v[1] != 0):
                    # find the value after the transformation
                    # du=float(small[0][1]*v[0])
                    # dv=float(small[0][2]*v[1])
                    # if(du>=0 and du<im2.shape[0] and dv>=0 and dv<im2.shape[1]):
                    # o = [small[0][2], small[0][1]]
                    # n = [dv, du]
                    du= int(i * v[0])
                    dv = int(j * v[1])
                    # du = int(i * v[1])
                    # dv = int(j * v[0])
                    n = [dv, du]
                    o=[j,i]
                    # add=0
                    # if(n[0]>=0 or n[1]>=0 or n[0]<im2.shape[1] or n[1]<im2.shape[0]):
                    #     add=1
                    # # if abs(dv - small[0][2]) > 625 and abs(du - small[0][1]) > 625:
                    # #     print("o =", o, "n=", n)
                    # #     add=0
                    # if add==1:
                    origpoints = np.append(origpoints, o)
                    newpoints = np.append(newpoints,n)


            except:
                z=z+1
                # print("caught exceptain")

    origpoints = origpoints.reshape(int(origpoints.shape[0] / 2),2)
    newpoints = newpoints.reshape(int(newpoints.shape[0] / 2), 2)

    print("origpoints" , origpoints[0],origpoints[1])
    print("newpoints",newpoints[0],newpoints[0])
    print("caught  %d exceptions" %(z))
    return origpoints , newpoints



def createPyramids(img:np.ndarray,levels :int):
    plist=[]
    k=cv2.getGaussianKernel(5,-1)
    ker = (k).dot(k.T)
    imgc=img.copy()
    for i in range(levels):
        imgc = cv2.filter2D(imgc,-1,ker,borderType=cv2.BORDER_REPLICATE)
        newr = np.floor(imgc.shape[0]/2).astype(int)
        newc = np.floor(imgc.shape[1]/2).astype(int)
        print((newc, newr))
        imnew = np.zeros((newr,newc))
        for j in range(newr):
            for k in range(newc):
                imnew[j][k]=imgc[j*2][k*2]
        plist.append(imnew)
        imgc = imnew

    f, ax = plt.subplots(1, levels)
    for i in range(levels) :
        ax[i].imshow(plist[i])
    plt.show()

    return plist



def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    pass


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    # black and white img
    if(len(img.shape)==2):
        print("black and white")
        plist = []
        k = cv2.getGaussianKernel(5, -1)
        ker = (k).dot(k.T)
        imgc = img.copy()
        for i in range(levels):
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
            plist.append(imgblur)
            newr = np.floor(imgblur.shape[0] / 2).astype(int)
            newc = np.floor(imgblur.shape[1] / 2).astype(int)
            imnew = np.zeros((newr, newc))
            for j in range(newr):
                for k in range(newc):
                    imnew[j][k] = imgblur[j*2][k*2]
            imgc = imnew
        return plist

    # color image
    else:
        print("color_img")
        plist = []
        k = cv2.getGaussianKernel(5, -1)
        ker = (k).dot(k.T)
        imgc = img.copy()
        for i in range(levels):
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
            plist.append(imgblur)
            newr = np.floor(imgblur.shape[0] / 2).astype(int)
            newc = np.floor(imgblur.shape[1] / 2).astype(int)
            imnew = np.zeros((newr, newc,3))
            for l in range(3):
                for j in range(newr):
                    for k in range(newc):
                        imnew[j][k][l] = imgblur[j*2][k*2][l]
            imgc = imnew
        return plist


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    plist = []
    k = cv2.getGaussianKernel(5, -1)
    ker = (k).dot(k.T)
    imgc = img.copy()
    # black and white image
    if len(img.shape)==2:
        print("black and white")
        for i in range(levels):
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
            lapimg=(imgc-imgblur)
            plist.append(lapimg)
            newr = np.floor(imgc.shape[0] / 2).astype(int)
            newc = np.floor(imgc.shape[1] / 2).astype(int)
            imnew = np.zeros((newr, newc))
            for j in range(newr):
                for k in range(newc):
                    imnew[j][k] = imgc[j * 2][k * 2]
            if i == levels-1:
                plist.append(imgc)
            imgc = imnew
        return plist


    # color image
    else:
        print("color_img")
        for i in range(levels):
            imgblur= cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
            imglap=(imgc-imgblur)
            plist.append(imglap)
            newr = np.floor(imgblur.shape[0] / 2).astype(int)
            newc = np.floor(imgblur.shape[1] / 2).astype(int)
            imnew = np.zeros((newr, newc, 3))
            for l in range(3):
                for j in range(newr):
                    for k in range(newc):
                        imnew[j][k][l] = imgblur[j * 2][k * 2][l]
            if i==levels-1:
                plist.append(imgc)
            imgc = imnew
        return plist



def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    k = cv2.getGaussianKernel(5, -1)
    ker = (k).dot(k.T)
    ker = ker * 4
    imgn = lap_pyr[-1]
    if (len(lap_pyr[-1].shape) == 2):
        print("black and white")
        for i in range(len(lap_pyr)-2,0,-1):
            print(i)
            imgn=imgn+lap_pyr[i]
            newr = lap_pyr[i - 1].shape[0]
            newc = lap_pyr[i - 1].shape[1]
            newimg=np.zeros((newr,newc))
            for j in range(imgn.shape[0]+2):
                for k in range(imgn.shape[1]+2):
                    if j*2 < newr and k*2 < newc:
                        j1=j
                        k1=k
                        if j >= imgn.shape[0]:
                            print("j is big")
                            j = imgn.shape[0]-1
                        if k >= imgn.shape[1]:
                            print("k is big")
                            k=imgn.shape[1]-1
                        newimg[j*2][k*2]=imgn[j1][k1]

            imgn= cv2.filter2D(newimg, -1, ker, borderType=cv2.BORDER_REPLICATE)
            # imgn=cv2.GaussianBlur(newimg,(5,5),cv2.BORDER_DEFAULT)
        imgn = imgn + lap_pyr[0]
        return imgn

    else:
        print("color img")
        for i in range(len(lap_pyr) - 2, 0, -1):
            imgn = imgn + lap_pyr[i]
            newr=lap_pyr[i - 1].shape[0]
            newc = lap_pyr[i - 1].shape[1]
            newimg = np.zeros((newr, newc,3))
            for l in range(3):
                for j in range(imgn.shape[0]+1):
                    for k in range(imgn.shape[1]+1):
                        if j * 2 < newr and k * 2 < newc :
                            j1 = j
                            k1 = k
                            if j >= imgn.shape[0]:
                                # print("j is big")
                                j1 = imgn.shape[0] - 1
                            if k >= imgn.shape[1]:
                                # print("k is big")
                                k1 = imgn.shape[1] - 1
                            newimg[j * 2][k * 2] = imgn[j1][k1]
            imgn = cv2.filter2D(newimg, -1, ker, borderType=cv2.BORDER_REFLECT_101)
        imgn=imgn+lap_pyr[0]
        return imgn


def pyrmask(mask: np.ndarray, levels: int) -> np.ndarray:
    plist = []
    k = cv2.getGaussianKernel(5, -1)
    ker = (k).dot(k.T)
    imgc = mask.copy()
    if (len(mask.shape) == 2):
        print("black and white")
        for i in range(levels):
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
            plist.append(imgblur)
            newr = np.floor(imgblur.shape[0] / 2).astype(int)
            newc = np.floor(imgblur.shape[1] / 2).astype(int)
            imnew = np.zeros((newr, newc))
            for j in range(newr):
                for k in range(newc):
                    imnew[j][k] = imgblur[j * 2][k * 2]
            if i == levels - 1:
                plist.append(imgc)
            imgc = imnew
        return plist

        # color image
    else:
        print("color_img")
        for i in range(levels):
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
            plist.append(imgblur)
            newr = np.floor(imgblur.shape[0] / 2).astype(int)
            newc = np.floor(imgblur.shape[1] / 2).astype(int)
            imnew = np.zeros((newr, newc, 3))
            for l in range(3):
                for j in range(newr):
                    for k in range(newc):
                        imnew[j][k][l] = imgblur[j * 2][k * 2][l]
            if i == levels - 1:
                plist.append(imgc)
            imgc = imnew
        return plist



def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    l1=laplaceianReduce(img_1,levels)
    l2=laplaceianReduce(img_2,levels)
    l3=laplaceianReduce(mask,levels)
    l5=pyrmask(mask,levels)
    # f, ax = plt.subplots(1, levels+1)
    # for i in range(levels+1):
    #     ax[i].imshow(l5[i])
    # plt.show()
    l4 = []

    for i in range(levels+1):
        # plt.imshow(1 - l3[i])
        # plt.show
        # plt.imshow((l3[i]))
        # plt.show
        l4.append((l5[i])*l1[i]+(1-l5[i])*l2[i])
        # plt.imshow(l4[-1])
        # plt.show
        # print(l1[i].shape, l2[i].shape, l3[i].shape, l4[i].shape, l5[i].shape)

    # f, ax = plt.subplots(1, 4)
    # for i in range(levels+1):
    #     plt.imshow(l4[i])
    #     plt.show()
        # ax[0].imshow(l4[i])
        # ax[1].imshow(l2[i])
        # ax[2].imshow(l3[i])
        # ax[3].imshow(l4[i])
        # plt.show()
    blended1=NormalizeData(laplaceianExpand(l4))
    naiveblend=NormalizeData(mask*img_1+(1-mask)*img_2)

    return naiveblend, blended1

