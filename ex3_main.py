import cv2
import numpy as np

from ex3_utils import *
import time


def MSE(a: np.ndarray, b: np.ndarray) -> float:
    return np.square(a - b).mean()
# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, (img_1.shape[1], img_1.shape[0]))


    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float), img_2.astype(np.float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv,0))
    print(np.mean(uv,0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Hierarchical LK Demo")

    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    im2 = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))

    ans = opticalFlowPyrLK(im1.astype(np.float), im2.astype(np.float), 4, 20, 5)

    pts = np.array([])
    uv = np.array([])
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                uv = np.append(uv, ans[i][j][0])
                uv = np.append(uv, ans[i][j][1])
                pts = np.append(pts, j)
                pts = np.append(pts, i)
    pts = pts.reshape(int(pts.shape[0] / 2), 2)
    uv = uv.reshape(int(uv.shape[0] / 2), 2)
    print(np.median(uv, 0))
    print(np.mean(uv, 0))
    displayOpticalFlow(im2, pts, uv)

    # problem with code
    # lk_params = dict(winSize=(15, 15),
    #                  maxLevel=2,
    #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    #                            10, 0.03))
    # p1, st, err=cv2.calcOpticalFlowPyrLK(np.float32(im1), np.float32(im2), pts,None, **lk_params)


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK")

    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -0.2],
                  [0, 1, -0.1],
                  [0, 0, 1]], dtype=np.float)
    im2 = cv2.warpPerspective(im1, t,(im1.shape[1],im1.shape[0]))

    pts,uv=opticalFlow(im1.astype(np.float), im2.astype(np.float), step_size=20, win_size=5)




    ans = opticalFlowPyrLK(im1.astype(np.float), im2.astype(np.float), 4, 20, 5)
    ptspyr = np.array([])
    uvpyr = np.array([])
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                uvpyr = np.append(uvpyr, ans[i][j][0])
                uvpyr = np.append(uvpyr, ans[i][j][1])
                ptspyr = np.append(ptspyr, j)
                ptspyr = np.append(ptspyr, i)
    ptspyr = ptspyr.reshape(int(ptspyr.shape[0] / 2), 2)
    uvpyr = uvpyr.reshape(int(uvpyr.shape[0] / 2), 2)
    if len(im2.shape)==2:
        f, ax = plt.subplots(1,3)
        ax[0].set_title('reg LK')
        ax[0].imshow(im2, cmap="gray")
        ax[0].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
        ax[1].set_title('Pyr LK')
        ax[1].imshow(im2, cmap="gray")
        ax[1].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='r')
        ax[2].set_title('overlap')
        ax[2].imshow(im2, cmap="gray")
        ax[2].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
        ax[2].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='y')
        plt.show()

    else:
        f, ax = plt.subplots(1, 3)
        ax[0].set_title('reg LK')
        ax[0].imshow(im2)
        ax[0].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
        ax[1].set_title('Pyr LK')
        ax[1].imshow(im2)
        ax[1].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='r')
        ax[2].set_title('overlap')
        ax[2].imshow(im2)
        ax[2].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
        ax[2].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='y')
        plt.show()


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    if len(img.shape)==2:
        plt.imshow(img, cmap='gray')
        plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
        plt.show()
    else:
        plt.imshow(img)
        plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
        plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------




def translationlkdemo(img_path):
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.4],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t,  (img_1.shape[1],img_1.shape[0]))
    st = time.time()
    mat=findTranslationLK(img_1, img_2)
    et = time.time()
    print("Time: {:.4f}".format(et - st))
    print("mat\n",mat ,"\nt\n", t)
    new = cv2.warpPerspective(img_1, mat, (img_1.shape[1],img_1.shape[0]))
    f, ax = plt.subplots(1, 3)
    ax[0].set_title('img2 given transformation')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('img2 found transformation')
    ax[1].imshow(new, cmap='gray')

    ax[2].set_title('diff')
    ax[2].imshow(img_2 - new, cmap='gray')
    print("mse= ", MSE(new, img_2))
    plt.show()


def rigidlkdemo(img_path):
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    t1 = np.array([[1, 0, 2],
                  [0, 1, -5],
                  [0, 0, 1]], dtype=np.float)
    theta=0.5
    t2=np.array([[np.cos(theta),-np.sin(theta),0],
                 [np.sin(theta),np.cos(theta),0],
                 [0,0,1]],dtype=np.float)

    t=t1@t2
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    mat = findRigidLK(img_1, img_2)
    et = time.time()
    print("Time: {:.4f}".format(et - st))
    print("mat\n", mat, "\nt\n", t)

    new = cv2.warpPerspective(img_1, mat, img_1.shape[::-1])
    f, ax = plt.subplots(1, 3)
    ax[0].set_title('img2 given transformation')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('img2 found transformation')
    ax[1].imshow(new, cmap='gray')

    ax[2].set_title('diff')
    ax[2].imshow(img_2 - new, cmap='gray')

    plt.show()
    print("mse= ", MSE(new, img_2))

def translationcorrdemo(img_path):
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, 5.5],
                  [0, 1, -2.75],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    mat =findTranslationCorr(img_1.astype(np.float),img_2.astype(np.float))
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print("mat\n", mat, "\nt\n", t)
    new = cv2.warpPerspective(img_1, mat, img_1.shape[::-1])
    f, ax = plt.subplots(1, 3)
    ax[0].set_title('img2 given transformation')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('img2 found transformation')
    ax[1].imshow(new, cmap='gray')

    ax[2].set_title('diff')
    ax[2].imshow(img_2-new, cmap='gray')

    plt.show()


def rigidcorrdemo(img_path):
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t1 = np.array([[1, 0, 2],
                  [0, 1, -2],
                  [0, 0, 1]], dtype=np.float)
    theta=0.05
    t2 = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]], dtype=np.float)
    t=t1@t2
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    mat = findRigidCorr(img_1.astype(np.float), img_2.astype(np.float))
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print("mat\n", mat, "\nt\n", t)
    new = cv2.warpPerspective(img_1, mat, img_1.shape[::-1])
    f, ax = plt.subplots(1, 3)
    ax[0].set_title('img2 given transformation')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('img2 found transformation')
    ax[1].imshow(new, cmap='gray')

    ax[2].set_title('diff')
    ax[2].imshow(img_2 - new, cmap='gray')

    plt.show()
    print("mse= ", MSE(new,img_2))

def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")
    #forward
    # img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    # img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    # t = np.array([[1, 0, 5.75],
    #                [0, 1, -3.25],
    #                [0, 0, 1]], dtype=np.float)
    # theta=0.1
    # # t2 = np.array([[np.cos(theta), -np.sin(theta), 0],
    # #                [np.sin(theta), np.cos(theta), 0],
    # #                [0, 0, 1]],dtype=np.float)
    # # t = t @ t2
    # img_2=cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    # new = np.zeros((img_1.shape[0],img_1.shape[1]))
    # st = time.time()
    # im2 = warpImages(img_1.astype(np.float),new.astype(np.float),t)
    # et = time.time()
    # print("Time: {:.4f}".format(et - st))
    # f, ax = plt.subplots(1, 3)
    # ax[0].set_title('my warp')
    # ax[0].imshow(im2,cmap="gray")
    #
    # ax[1].set_title('cv2 warp')
    # ax[1].imshow(img_2,cmap="gray")
    #
    # ax[2].set_title('diff')
    # ax[2].imshow(img_2 - im2,cmap="gray")
    # plt.show()
    # print("mse= ",MSE(img_2,im2))


    # #backward
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -0.2],
                  [0, 1, 0.2],
                  [0, 0, 1]], dtype=np.float)
    theta=0
    t2 = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]],dtype=np.float)
    t = t @ t2
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    im2 = warpImages(img_1.astype(np.float), img_2.astype(np.float), t)
    et = time.time()
    print("Time: {:.4f}".format(et - st))
    f, ax = plt.subplots(1, 3)
    ax[0].set_title('my rewarp')
    ax[0].imshow(im2)

    ax[1].set_title('cv2 warp')
    ax[1].imshow(img_1)

    ax[2].set_title('diff')
    ax[2].imshow(img_2 - im2)
    plt.show()
    print("mse= ", MSE(img_2, im2))

# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))
    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]


    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()



def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    # cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'


    # lkDemo(img_path)
    # hierarchicalkDemo(img_path)
    # compareLK(img_path)
    translationlkdemo(img_path)
    # translationcorrdemo(img_path)
    # rigidcorrdemo(img_path)
    # imageWarpingDemo(img_path)
    # rigidlkdemo(img_path)

    # # make a new warped image
    # img_path = 'input/color1.jpg'
    # img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    # t = np.array([[1, 0, 10],
    #               [0, 1, -5],
    #               [0, 0, 1]], dtype=np.float)
    # img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    # plt.imshow(img_2)
    # plt.show()
    # cv2.imwrite("Input/color3.jpg", img_2)


    # pyrGaussianDemo('input/pyr_bit.jpg')
    # pyrLaplacianDemo('input/pyr_bit.jpg')
    # blendDemo()



if __name__ == '__main__':
    main()
