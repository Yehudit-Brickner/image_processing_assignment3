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
    # make the derivative of x y and diff
    Ix = cv2.filter2D(im2, -1, ker, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, ker.T, borderType=cv2.BORDER_REPLICATE)
    It=im2-im1

    rep = win_size//2

    origpoints = np.array([])
    newpoints = np.array([])
    start= int(np.floor(win_size/2))
    # running over the images in steps of step_size =  making a grid of the image
    # and taking the middle of the grid to be the middle of the "window" later on
    for i in range(start,im1.shape[0]-start, step_size):
        for j in range(start, im1.shape[1]-start,step_size):

            # getting the "window" of parts of the images

            ATA=np.array([[(Ix[i-rep : i+rep+1, j-rep : j+rep+1] * Ix[i-rep : i+rep+1, j-rep : j+rep+1]).sum(), (Ix[i-rep : i+rep+1, j-rep : j+rep+1] * Iy[i-rep : i+rep+1, j-rep : j+rep+1]).sum()],
                [(Ix[i-rep : i+rep+1, j-rep : j+rep+1] * Iy[i-rep : i+rep+1, j-rep : j+rep+1]).sum(), (Iy[i-rep : i+rep+1, j-rep : j+rep+1] * Iy[i-rep : i+rep+1, j-rep : j+rep+1]).sum()]])

            ATB = np.array([(Ix[i-rep : i+rep+1, j-rep : j+rep+1] * It[i-rep : i+rep+1, j-rep : j+rep+1]).sum(), (Iy[i-rep : i+rep+1, j-rep : j+rep+1] * It[i-rep : i+rep+1, j-rep : j+rep+1]).sum()])
            # ATB=ATB*-1

            # getting eigen values and sorting them
            e, e1 = np.linalg.eig(ATA)
            e = np.sort(e)
            # make sure the eigen values are ok
            if  e[0] > 1 and e[1]/e[0] < 100:
                vec = np.linalg.inv(ATA) @ (ATB)

                n = [vec[0], vec[1]]
                o=[j,i]
                # adding the points and u,v to there respective arrays
                origpoints = np.append(origpoints, o)
                newpoints = np.append(newpoints,n)


    # changing the shape of the array so that it is in pairs of x,y / u,v
    origpoints = origpoints.reshape(int(origpoints.shape[0] / 2),2)
    newpoints = newpoints.reshape(int(newpoints.shape[0] / 2), 2)

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
    # create a pyramid of im1 and im2 with k levels - the first being the original image
    A=[]
    B=[]
    A.append(img1)
    B.append(img2)

    for i in range(k-1):
        A.append(cv2.pyrDown(A[-1], (A[-1].shape[0] // 2, A[-1].shape[1] // 2)))
        B.append(cv2.pyrDown(B[-1], (B[-1].shape[0] // 2, B[-1].shape[1] // 2)))

    # create a pyramid of the change in a (m,n,2) matrix per level
    C = []
    for i in range(len(A)):
        a = A[i].shape[0]
        b = A[i].shape[1]
        change=np.zeros((a,b,2))
        old, new = opticalFlow(A[i], B[i], step_size=stepSize, win_size=winSize)
        for x in range(len(old)):
            b=old[x][0].astype(int) # height of pix
            a=old[x][1].astype(int) # width of pix
            c=new[x][0] # u of transformation
            d=new[x][1] # v of transformation
            change[a][b][0]=c
            change[a][b][1]=d
        C.append(change)

    # start from the smallest mat in c and add it into the next level
    # by making the x,y bigger by 2 and the value bigger by 2
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

    diff = float("inf")
    spot = 0
    old, new=opticalFlow(im1,im2,10,5)
    # look at all the u,v we found
    for x in range(len(new)):
        t1=new[x][0]
        t2=new[x][1]
        t = np.array([[1, 0, t1],
                      [0, 1, t2],
                      [0, 0, 1]], dtype=np.float)
        # create a new image a transformation using u,v
        newimg = cv2.warpPerspective(im1, t, (im1.shape[1],im1.shape[0]))
        # find difference in image and keep track of the x,y that gives the smallest diff
        d= ((im2-newimg)**2).sum()

        if d<diff:
            diff = d
            spot = x
            if diff==0:
                print("break")
                break
    t1=new[spot][0]
    t2=new[spot][1]

    t = np.array([[1, 0, t1],
                  [0, 1, t2],
                  [0, 0, 1]], dtype=np.float)

    return t


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """

    old, new = opticalFlow(im1, im2, 10, 5)
    diff= float("inf")
    # look at all the u,v we found
    for n in range(new.shape[0]):
        x =  new[n][0]
        y = new[n][1]
        # find angel between the points
        if(x!=0):
            theta= np.arctan(y / x)
        else:
            theta =0
        # create a new image a transformation using u,v and theta
        t = np.array([[np.cos(theta), -np.sin(theta), x],
                       [np.sin(theta), np.cos(theta), y],
                       [0, 0, 1]], dtype=np.float)
        newpic= cv2.warpPerspective(im1, t,(im1.shape[1], im1.shape[0]))
        # find difference in image and keep track of the x,y, theta that gives the smallest diff
        d = ((im2 - newpic) ** 2).sum()
        if d<diff:
            diff=d
            spot=n
        if diff==0:
            break

    x = new[spot][0]
    y = new[spot][1]

    if x!=0:
        theta = np.arctan(y/x)
    else:
        theta=0
    t = np.array([[np.cos(theta), -np.sin(theta), x],
                  [np.sin(theta), np.cos(theta), y],
                  [0, 0, 1]], dtype=np.float)

    return (t)



def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    win =13
    pad= win//2
    im2pad = cv2.copyMakeBorder(im2, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)

    # getting 4 x and y points to be the middle of the window
    # the points are 1/5, 2/5 ... of the length and height
    I=[]
    J=[]
    for x in range (1,5):
        I.append((im1.shape[0]//5)*x)
        J.append((im1.shape[1]//5)*x)


    corr_listt=[(np.array([0]),0,0)]
    for x in range(len(I)):
        for y in range(len(J)):
            # getting a template to match
            windowa = im1[I[x] - pad:I[x] + pad + 1, J[y] - pad:J[y] + pad + 1]
            a = windowa.reshape(1, win * win)
            aT = a.T
            big = [(np.array([0]), 0, 0)]
            # going through the other pic to match the template
            for i in range(0,im2.shape[0]):
                for j in range(0,im2.shape[1]):
                    if (i+pad+win)<im2pad.shape[0] and (j+pad+win)<im2pad.shape[1] :
                        windowb= im2pad[i+pad:i+pad+win, j+pad:j+pad+win]
                        b = windowb.reshape(1, win * win)
                        bT=b.T
                        top = np.dot(a, bT)
                        bottom = np.dot(a, aT) + np.dot(b, bT)
                        # finding the correlation between the template and this window
                        # if it is bigger than the first value in list big clear big and put it in with the x y values of im2
                        # if it is equal to the first value add it to the list and put it in with the x y values of im2
                        if bottom != 0:
                            corr = top / bottom
                            if corr > big[0][0]:
                                big.clear()
                                big.insert(0, (corr, i, j))
                            elif corr == big[0][0]:
                                big.insert(0, (corr, i, j))
            # after checking this template check if the first value in big is bigger than the first value in corr_lisst
            # if so clear corr_listt and copy the values from big to corr_listt and add the x y vaues of the original image
            # if it equals copy the values from big to corr_listt and add the x y vaues of the original image
            if big[0][0][0] > corr_listt[0][0][0]:
                corr_listt.clear()
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))
            if big[0][0][0] == corr_listt[0][0][0]:
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))

    dif=float("inf")
    spot=-1
    # go through all values in the cor list and find the u v by finding the difference between im1 xy and im2 xy
    for x in range (len(corr_listt)):

        t1 = corr_listt[x][1][0] - corr_listt[x][0][1] # u
        t2 = corr_listt[x][1][1] - corr_listt[x][0][2] # v
    # create a new img with the found transformation
        t = np.array([[1, 0, t1],
                      [0, 1, t2],
                      [0, 0, 1]], dtype=np.float)
        new=cv2.warpPerspective(im1, t, (im1.shape[1],im1.shape[0]))
        # find the difference between new and im2 if smaller than diff update diff and spot
        d= ((im2-new)**2).sum()
        if d<dif:
            dif=d
            spot=x
            if dif==0:
                break
    # take the values from corrlist that has the smallest diff and return the transformation
    t1 = corr_listt[spot][1][0] - corr_listt[spot][0][1] # u
    t2 = corr_listt[spot][1][1] - corr_listt[spot][0][2] # v
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
    win = 5
    pad = win // 2

    im2pad = cv2.copyMakeBorder(im2, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)

    # getting 4 x and y points to be the middle of the window
    # the points are 1/5, 2/5 ... of the length and height
    I = []
    J = []
    for x in range(1, 5):
        I.append((im1.shape[0] // 5) * x)
        J.append((im1.shape[1] // 5) * x)

    corr_listt = [(np.array([0]), 0, 0)]
    for x in range(len(I)):
        for y in range(len(J)):
            # getting a template to match
            windowa = im1[I[x] - pad:I[x] + pad + 1, J[y] - pad:J[y] + pad + 1]
            a = windowa.reshape(1, win * win)
            aT = a.T
            big = [(np.array([0]), 0, 0)]
            for i in range(0, im2.shape[0]):
                for j in range(0, im2.shape[1]):
                    if (i + pad + win) < im2pad.shape[0] and (j + pad + win) < im2pad.shape[1]:
                        windowb = im2pad[i + pad:i + pad + win, j + pad:j + pad + win]
                        b = windowb.reshape(1, win * win)
                        bT = b.T
                        top = np.dot(a, bT)
                        bottom = np.dot(a, aT) + np.dot(b, bT)
                        # finding the correlation between the template and this window
                        # if it is bigger than the first value in list big clear big and put it in with the x y values of im2
                        # if it is equal to the first value add it to the list and put it in with the x y values of im2
                        if bottom != 0:
                            corr = top / bottom
                            if corr > big[0][0]:
                                big.clear()
                                big.insert(0, (corr, i, j))
                            elif corr == big[0][0]:
                                big.insert(0, (corr, i, j))
            # after checking this template check if the first value in big is bigger than the first value in corr_lisst
            # if so clear corr_listt and copy the values from big to corr_listt and add the x y vaues of the original image
            # if it equals copy the values from big to corr_listt and add the x y vaues of the original image
            if big[0][0][0] > corr_listt[0][0][0]:
                corr_listt.clear()
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))
            if big[0][0][0] == corr_listt[0][0][0]:
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))

    spot=-1
    diff=float("inf")
    # go through all values in the cor_list and find the u v and theta
    # by finding the difference between im1 xy and im2 xy
    for n in range(len(corr_listt)):
        x =  corr_listt[n][1][0] - corr_listt[n][0][1]
        y = corr_listt[n][1][1] - corr_listt[n][0][2]

        if(y!=0):
            theta= np.arctan(x/ y)
        else:
            theta =0
        # create a new img with the found transformation
        t = np.array([[np.cos(theta), -np.sin(theta), x],
                       [np.sin(theta), np.cos(theta), y],
                       [0, 0, 1]], dtype=np.float)
        new= cv2.warpPerspective(im1, t,im1.shape[::-1])
        # find the difference between new and im2 if smaller than diff update diff and spot
        d = ((im2 - new) ** 2).sum()
        if d<diff:
            diff=d
            spot=n
        if diff==0:
            break
    # take the values from corrlist that has the smallest diff and return the transformation
    x = corr_listt[spot][1][0] - corr_listt[spot][0][1]
    y = corr_listt[spot][1][1] - corr_listt[spot][0][2]

    if y!=0:
        theta = np.arctan(x /y)
    else:
        theta=0
    t = np.array([[np.cos(theta), -np.sin(theta), x],
                  [np.sin(theta), np.cos(theta), y],
                  [0, 0, 1]], dtype=np.float)

    return (t)





def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """

    new=np.zeros((im1.shape[0],im1.shape[1]))
    Tinv = np.linalg.inv(T)
    print(T, "\n",Tinv)
    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            arr= np.array([i,j,1])
            newarr=Tinv@arr
            x1 = np.floor(newarr[0]).astype(int)
            x2 = np.ceil(newarr[0]).astype(int)
            x3= round(newarr[0]%1, 3)
            y1 = np.floor(newarr[1]).astype(int)
            y2 = np.ceil(newarr[1]).astype(int)
            y3 = round(newarr[1]%1,3)


            if x1>=0 and y1>=0 and x1<im1.shape[0] and y1<im1.shape[1]:
                new[i][j]+= (1-x3) * (1-y3) * im1[x1][y1]

            if x2 >= 0 and y1 >= 0 and x2 < im1.shape[0] and y1 < im1.shape[1]:
                new[i][j] += x3 * (1-y3) * im1[x2][y1]

            if x1 >= 0 and y2 >= 0 and x1 < im1.shape[0] and y2 < im1.shape[1]:
                new[i][j] += (1-x3) * y3 * im1[x1][y2]

            if x2 >= 0 and y2 >= 0 and x2 < im1.shape[0] and y2 < im1.shape[1]:
                new[i][j] += x3 * y3 * im1[x2][y2]


    plt.imshow(new)
    plt.show()
    return new
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

