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



def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
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
    Ix = cv2.filter2D(im2, -1, ker, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, ker.T, borderType=cv2.BORDER_REPLICATE)
    It=im2-im1
    z=0
    origpoints = np.array([])
    newpoints = np.array([])
    start= int(np.floor(win_size/2))

    for i in range(start,im1.shape[0]-start, step_size):
        for j in range(start, im1.shape[1]-start,step_size):
            try:
                # find the u and v for the best matched area
                B = -It[i-rep : i+rep+1, j-rep : j+rep+1]
                B = B.reshape(win_size**2, 1)
                A1 = Ix[i-rep : i+rep+1, j-rep : j+rep+1].flatten()
                A2 = Iy[i-rep : i+rep+1, j-rep : j+rep+1].flatten()
                # A = np.stack((A1,A2))
                A = np.array([])
                for x in range(len(A1)):
                    A = np.append(A,A1[x])
                    A = np.append(A,A2[x])
                A = A.reshape(win_size**2,2)
                AT = A.T
                ATA = AT@A

                # ATA=np.array([[(Ix[i-rep : i+rep+1, j-rep : j+rep+1] * Ix[i-rep : i+rep+1, j-rep : j+rep+1]).sum(), (Ix[i-rep : i+rep+1, j-rep : j+rep+1] * Iy[i-rep : i+rep+1, j-rep : j+rep+1]).sum()],
                #     [(Ix[i-rep : i+rep+1, j-rep : j+rep+1] * Iy[i-rep : i+rep+1, j-rep : j+rep+1]).sum(), (Iy[i-rep : i+rep+1, j-rep : j+rep+1] * Iy[i-rep : i+rep+1, j-rep : j+rep+1]).sum()]])
                #
                # ATB = np.array([(Ix[i-rep : i+rep+1, j-rep : j+rep+1] * It[i-rep : i+rep+1, j-rep : j+rep+1]).sum(), (Iy[i-rep : i+rep+1, j-rep : j+rep+1] * It[i-rep : i+rep+1, j-rep : j+rep+1]).sum()])
                # ATB=ATB*-1
                e, e1 = np.linalg.eig(ATA)
                e = np.sort(e)
                # make sure the eigen values are ok
                if  e[0] > 1 and e[1]/e[0] < 100:
                    v = np.linalg.inv(ATA) @ -(AT @ B)
                    # v = np.linalg.inv(ATA) @ (ATB)
                    # n = [v[0], v[1]]
                    n = [v[0], v[1]]
                    o=[j,i]
                    origpoints = np.append(origpoints, o)
                    newpoints = np.append(newpoints,n)


            except:
                z=z+1
                # print("caught exceptain")

    origpoints = origpoints.reshape(int(origpoints.shape[0] / 2),2)
    newpoints = newpoints.reshape(int(newpoints.shape[0] / 2), 2)

    # print("origpoints" , origpoints[0],origpoints[1])
    # print("newpoints",newpoints[0],newpoints[0])
    print("caught  %d exceptions" %(z))
    return origpoints , newpoints




def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int, stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    A=[]
    B=[]
    A.append(img1)
    B.append(img2)

    for i in range(k-1):
        A.append(cv2.pyrDown(A[-1], (A[-1].shape[0] // 2, A[-1].shape[1] // 2)))
        B.append(cv2.pyrDown(B[-1], (B[-1].shape[0] // 2, B[-1].shape[1] // 2)))

    C = []
    for i in range(len(A)):
        a = A[i].shape[0]
        b = A[i].shape[1]
        change=np.zeros((a,b,2))
        old, new = opticalFlow(A[i], B[i], step_size=stepSize, win_size=winSize)
        for x in range(len(old)):
            b=old[x][0].astype(int) # height of pix
            a=old[x][1].astype(int) # width of pix
            c=new[x][0] # u of tran
            d=new[x][1] # v of tran
            change[a][b][0]=c
            change[a][b][1]=d
        C.append(change)

    for x in range(-1,-k,-1):
        y = x - 1
        for i in range(C[x].shape[0]):
            for j in range(C[x].shape[1]):
                C[y][i*2][j*2][0] += (C[x][i][j][0]*2)
                C[y][i*2][j*2][1] += (C[x][i][j][1]*2)

    return C[0]




# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    listt=[]
    old, new=opticalFlow(im1,im2,10,5)
    for x in range(len(new)):
        # print("new",new[x])
        t1=new[x][0]
        t2=new[x][1]
        # print(t1, t2)
        t = np.array([[1, 0, t1],
                      [0, 1, t2],
                      [0, 0, 1]], dtype=np.float)
        newimg = cv2.warpPerspective(im1, t, im1.shape[::-1])
        listt.append(((im2-newimg).sum(),t1,t2))
   # # check median
   #  m1 = np.median(new, 0)
   #  t1 = m1[0]
   #  t2 = m1[1]
   #  t = np.array([[1, 0, t1],
   #                [0, 1, t2],
   #                [0, 0, 1]], dtype=np.float)
   #  newimg = cv2.warpPerspective(im1, t, im1.shape[::-1])
   #  listt.append(((im2 - newimg).sum(), t1, t2))
   #  # check mean
   #  m1 = np.mean(new, 0)
   #  t1 = m1[0]
   #  t2 = m1[1]
   #  t = np.array([[1, 0, t1],
   #                [0, 1, t2],
   #                [0, 0, 1]], dtype=np.float)
   #  newimg = cv2.warpPerspective(im1, t, im1.shape[::-1])
   #  listt.append(((im2 - newimg).sum(), t1, t2))

    y = float("inf")
    spot=0
    for x in range(len (listt)):
        if listt[x][0]<y:
            y=listt[x][0]
            spot=x
        if y==0:
            break

    t1=listt[spot][1]
    t2=listt[spot][2]
    t = np.array([[1, 0, t1],
                  [0, 1, t2],
                  [0, 0, 1]], dtype=np.float)

    # print(t)
    return t


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    listt = []
    old, new = opticalFlow(im1, im2, 10, 5)


    pass



def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    win =7
    pad= win//2
    im1pad = cv2.copyMakeBorder(im1, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)
    im2pad = cv2.copyMakeBorder(im2, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)
    I=[]
    J=[]
    for x in range (1,5):
        I.append((im1.shape[0]//5)*x)
        J.append((im1.shape[1]//5)*x)


    corr_listt=[(np.array([0]),0,0)]
    for x in range(len(I)):
        for y in range(len(J)):
            windowa = im1[I[x] - pad:I[x] + pad + 1, J[y] - pad:J[y] + pad + 1]
            aa=windowa.flatten()
            a = windowa.reshape(1, win * win)
            big = [(np.array([0]), 0, 0)]
            for i in range(0,im2.shape[0]):
                for j in range(0,im2.shape[1]):
                    if (i+pad+win)<im2pad.shape[0] and (j+pad+win)<im2pad.shape[1] :
                        windowb= im2pad[i+pad:i+pad+win, j+pad:j+pad+win]
                        bb=windowb.flatten()
                        b=windowb.reshape(1, win*win)
                        top=np.correlate(aa,bb)
                        bottom=np.correlate(aa,aa)+np.correlate(bb,bb)
                        corr =top/bottom
                        if corr > big[0][0]:
                            big.clear()
                            big.insert(0, (corr, i, j))
                        elif corr == big[0][0]:
                            big.insert(0, (corr, i, j))
            if big[0][0][0] > corr_listt[0][0][0]:
                corr_listt.clear()
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))
            if big[0][0][0] == corr_listt[0][0][0]:
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))

    dif=float("inf")
    spot=-1
    print(len(corr_listt))
    for x in range (len(corr_listt)):

        t1 = corr_listt[x][1][0] - corr_listt[x][0][1]
        t2 = corr_listt[x][1][1] - corr_listt[x][0][2]

        t = np.array([[1, 0, t1],
                      [0, 1, t2],
                      [0, 0, 1]], dtype=np.float)
        new=cv2.warpPerspective(im1, t, im1.shape[::-1])
        d= ((im2-new)**2).sum()
        if d<dif:
            dif=d
            spot=x
            if dif==0:
                break
    print("diff=", dif , "spot=", spot)

    t1 = corr_listt[spot][1][0] - corr_listt[spot][0][1]
    t2 = corr_listt[spot][1][1] - corr_listt[spot][0][2]
    t = np.array([[1, 0, t1],
                  [0, 1, t2],
                  [0, 0, 1]], dtype=np.float)
    return t


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
        # print("black and white")
        plist = []
        k = cv2.getGaussianKernel(5, -1)
        ker = (k).dot(k.T)
        imgc = img.copy()
        for i in range(levels):
            plist.append(imgc)
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
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
        # print("color_img")
        plist = []
        k = cv2.getGaussianKernel(5, -1)
        ker = (k).dot(k.T)
        imgc = img.copy()
        for i in range(levels):
            plist.append(imgc)
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
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
        # print("black and white")
        for i in range(levels):
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
            imglap = (imgc - imgblur)
            plist.append(imglap)
            newr = np.floor(imgblur.shape[0] / 2).astype(int)
            newc = np.floor(imgblur.shape[1] / 2).astype(int)
            imnew = np.zeros((newr, newc))
            for j in range(newr):
                for k in range(newc):
                    imnew[j][k] = imgblur[j * 2][k * 2]
            if i == levels-1:
                plist.append(imgc)
            imgc = imnew
        return plist


    # color image
    else:
        # print("color_img")
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
        # print("black and white")
        for i in range(len(lap_pyr)-2,0,-1):
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
                            j = imgn.shape[0]-1
                        if k >= imgn.shape[1]:
                            k=imgn.shape[1]-1
                        newimg[j*2][k*2]=imgn[j1][k1]

            imgn= cv2.filter2D(newimg, -1, ker, borderType=cv2.BORDER_REFLECT_101)
        imgn = imgn + lap_pyr[0]
        return imgn

    else:
        # print("color img")
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
                                j1 = imgn.shape[0] - 1
                            if k >= imgn.shape[1]:
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
        # print("black and white")
        for i in range(levels):
            plist.append(imgc)
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
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
        # print("color_img")
        for i in range(levels):
            plist.append(imgc)
            imgblur = cv2.filter2D(imgc, -1, ker, borderType=cv2.BORDER_REFLECT_101)
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



def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """

    k = cv2.getGaussianKernel(5, -1)
    ker = (k).dot(k.T)
    l1=laplaceianReduce(img_1,levels)
    l2=laplaceianReduce(img_2,levels)
    # l3=laplaceianReduce(mask,levels)
    l5=pyrmask(mask,levels)
    l4 = []

    for i in range(levels+1):
        l4.append((l5[i])*l1[i]+(1-l5[i])*l2[i])
    blended1=NormalizeData(laplaceianExpand(l4))
    naiveblend=NormalizeData(mask*img_1+(1-mask)*img_2)
    # naiveblend=cv2.filter2D(naiveblend, -1, ker, borderType=cv2.BORDER_REFLECT_101)
    return naiveblend, blended1

