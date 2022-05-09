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
    eps=0.05
    ker = np.array([[1, 0, -1]])
    rep = int(np.floor(win_size/2))
    # print(rep)
    im2pad = cv2.copyMakeBorder(im2, rep, rep, rep, rep, cv2.BORDER_REPLICATE, None, value=0)
    Ix = cv2.filter2D(im2pad, -1, ker, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2pad, -1, ker.T, borderType=cv2.BORDER_REPLICATE)
    It=im2-im1

    # f, ax = plt.subplots(1, 5)
    # ax[0].imshow(im1)
    # ax[1].imshow(Ix)
    # ax[2].imshow(Iy)
    # ax[3].imshow(im2)
    # ax[4].imshow(It)
    # plt.show()
    # print(im2.shape)
    origpoints = np.array([])
    newpoints = np.array([])
    for i in range( 3,im2.shape[0]-3,step_size):
        for j in range(3,im2.shape[1]-3,step_size):
            print("i j ", i ,j)
            try:
                A = np.array([])
                B = np.array([])
                for k in range(i-2,i+3):
                    for l in range(j-2,j+3):
                       A = np.append(A,Ix[k,l])
                       A = np.append(A,Iy[k, l])
                       B = np.append(B, -It[k, l])
                B = B.reshape(25,1)
                A = A.reshape(25,2)
                AT = A.T
                ATA=AT@A
                e, e1=np.linalg.eig(ATA)
                e=np.sort(e)

                if e[1]>=e[0]>1 and e[1]/e[0]<100:
                    # print(i, j)
                    # print("eigen values ", e)
                    v = np.linalg.inv(ATA) @ AT @ B

                    if(v[0]!=0 and v[1]!=0):
                        # print(v)
                        # print("we arent zero")
                        for k in range(i - 2, i + 3):
                            for l in range(j - 2, j + 3):
                                du=int(k+v[0])
                                dv=int(l+v[1])
                                if(du>=0 and du<im1.shape[0] and dv>=0 and dv<im1.shape[1]):
                                    # print([du,dv], [k,l])
                                    o = [k, l]
                                    n = [du, dv]
                                    add=1
                                    for m in range(-1,-50,-2):
                                       if len(newpoints)>-(m-1):
                                            if(newpoints[m]==du and newpoints[m-1]==dv):
                                                add=0
                                                print(du,dv)
                                    if add==1:
                                        newpoints = np.append(newpoints,n)
                                        origpoints = np.append(origpoints,o)
                                    # print([int(du),int(dv)])
            except:
                print("caught exceptain")
    origpoints = origpoints.reshape(int(origpoints.shape[0] / 2),2)
    newpoints = newpoints.reshape(int(newpoints.shape[0] / 2), 2)

    print("origpoints" , origpoints[0],origpoints[1])
    print("newpoints",newpoints[0],newpoints[0])
    return origpoints,newpoints



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



    pass


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
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REPLICATE)
            newr = np.ceil(imgblur.shape[0] / 2).astype(int)
            newc = np.ceil(imgblur.shape[1] / 2).astype(int)
            print((newc, newr))
            imnew = np.zeros((newr, newc))
            for j in range(newr):
                for k in range(newc):
                    imnew[j][k] = imgblur[j*2][k*2]
            plist.append(imnew)
            imgc = imnew

        f, ax = plt.subplots(1, len(plist))
        for i in range(len(plist)):
            ax[i].imshow(plist[i])
        plt.show()
        return plist

    # color image
    else:
        print("color_img")
        plist = []
        k = cv2.getGaussianKernel(5, -1)
        ker = (k).dot(k.T)
        imgc = img.copy()
        for i in range(levels):
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REPLICATE)
            newr = np.ceil(imgblur.shape[0] / 2).astype(int)
            newc = np.ceil(imgblur.shape[1] / 2).astype(int)
            print((newc, newr))
            imnew = np.zeros((newr, newc,3))
            for l in range(3):
                for j in range(newr):
                    for k in range(newc):
                        imnew[j][k][l] = imgblur[j*2][k*2][l]
            plist.append(imnew)
            imgc = imnew
        f, ax = plt.subplots(1, len(plist))
        for i in range(len(plist)):
            ax[i].imshow(plist[i])
        plt.show()
        return plist


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    # black and white image
    if(len(img.shape)==2):
        print("black and white")
        plist = []
        k = cv2.getGaussianKernel(5, -1)
        ker = (k).dot(k.T)
        imgc = img.copy()
        for i in range(levels):
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REPLICATE)
            lapimg=imgc-imgblur
            plist.append(lapimg)
            newr = np.ceil(imgc.shape[0] / 2).astype(int)
            newc = np.ceil(imgc.shape[1] / 2).astype(int)
            print((newc, newr))
            imnew = np.zeros((newr, newc))
            for j in range(newr):
                for k in range(newc):
                    imnew[j][k] = imgc[j * 2][k * 2]
            if i==levels-1:
                plist.append(imgc)
            imgc = imnew

        f, ax = plt.subplots(1, len(plist))
        for i in range(len(plist)):
            ax[i].imshow(plist[i])
        plt.show()
        return plist


    # color image
    else:
        print("color_img")

        plist = []
        k = cv2.getGaussianKernel(5, -1)
        ker = (k).dot(k.T)
        imgc = img.copy()
        for i in range(levels):
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REPLICATE)
            imglap=(imgc-imgblur)
            plist.append(imglap)
            newr = np.ceil(imgblur.shape[0] / 2).astype(int)
            newc = np.ceil(imgblur.shape[1] / 2).astype(int)
            # print((newc, newr))
            imnew = np.zeros((newr, newc, 3))
            for l in range(3):
                for j in range(newr):
                    for k in range(newc):
                        imnew[j][k][l] = imgblur[j * 2][k * 2][l]
            if i==levels-1:
                plist.append(imgc)
            imgc = imnew
        f, ax = plt.subplots(1, len(plist))
        for i in range(len(plist)):
            ax[i].imshow(plist[i])
        plt.show()
        return plist



def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    if (len(lap_pyr[-1].shape) == 2):
        print("black and white")
        plist = []
        k = cv2.getGaussianKernel(5, -1)
        ker = (k).dot(k.T)
        ker=ker*4
        imgn=lap_pyr[-1]
        for i in range(len(lap_pyr)-2,0,-1):
            print(i)
            imgn=imgn+lap_pyr[i]
            newr = (imgn.shape[0]*2)
            newc = (imgn.shape[1]*2)
            print(newr, newc)
            newimg=np.zeros((newr,newc))
            for j in range(imgn.shape[0]):
                for k in range(imgn.shape[1]):
                    newimg[j*2][k*2]=imgn[j][k]
            imgn= cv2.filter2D(newimg, -1, ker, borderType=cv2.BORDER_REPLICATE)
            plt.imshow(imgn)
            plt.show()
        imgn=imgn+lap_pyr[0]
        plt.imshow(imgn)
        plt.show()
        return imgn

    else:
        print("color img")
        plist = []
        k = cv2.getGaussianKernel(5, -1)
        ker = (k).dot(k.T)
        ker = ker * 4
        imgn = lap_pyr[-1]
        for i in range(len(lap_pyr) - 2, 0, -1):
            print(i, imgn.shape, lap_pyr[i].shape)
            imgn = imgn + lap_pyr[i]
            newr = (imgn.shape[0] * 2)
            newc = (imgn.shape[1] * 2)
            print(newr, newc)
            newimg = np.zeros((newr, newc,3))
            for l in range(3):
                for j in range(imgn.shape[0]):
                    for k in range(imgn.shape[1]):
                        newimg[j * 2][k * 2][l] = imgn[j][k][l]
            imgn = cv2.filter2D(newimg, -1, ker, borderType=cv2.BORDER_REPLICATE)
            plt.show(imgn)
            plt.show()
        imgn = imgn + lap_pyr[0]
        plt.imshow(imgn)
        plt.show()
        return imgn





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
    l4=[]
    for i in range(levels+1):
        l4.append(l3[i]*l1[i]+(1-l3[i])*l2[i])
        print(l1[i].shape, l2[i].shape, l3[i].shape, l4[i].shape)

    # f, ax = plt.subplots(1, 4)
    # for i in range(levels+1):
    #     plt.imshow(l4[i])
    #     plt.show()
        # ax[0].imshow(l4[i])
        # ax[1].imshow(l2[i])
        # ax[2].imshow(l3[i])
        # ax[3].imshow(l4[i])
        # plt.show()
    blended1=laplaceianExpand(l4)
    plt.imshow(blended1)
    plt.show()
    pass

