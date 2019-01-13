import numpy as np
import cv2


def specification_2d(img):
    # initialize gray_level distribution
    gray_level_prob = {}
    gray_level_dist = {}
    gray_map = {}
    for i in range(256):
        gray_level_prob[i] = 0
        gray_level_dist[i] = 0

    # calculate the gray_level_prob
    for pixel in img.flatten():
        gray_level_prob[pixel] += 1

    total_gray = 0
    for key in gray_level_prob:
        total_gray += gray_level_prob[key]
    print(total_gray)

    # calculate the gray_level_dist
    gray_level_dist[0] = gray_level_prob[0] / total_gray
    for i in range(1, 256):
        gray_level_dist[i] = gray_level_dist[i - 1] + gray_level_prob[i] / total_gray
    print(gray_level_prob)
    print(gray_level_dist)

    # regulation find gray_level mapping
    for i in range(256):
        mapping = int(gray_level_dist[i] * 256)
        gray_map[i] = mapping
    print(gray_map)
    print('-----------------------------------')
    shape = img.shape
    # reconstruction the graph
    img_new = np.array([gray_map[pixel] for pixel in img.flatten()])
    print(img_new.reshape(shape))
    return img_new.reshape(shape)/255


def pca_2d(image_2d):
    # FUNCTION FOR RECONSTRUCTING 2D MATRIX USING PCA
    cov_mat = image_2d - np.mean(image_2d)
    eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat))
    # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
    p = np.size(eig_vec, axis=1)
    idx = np.argsort(eig_val)
    idx = idx[::-1]
    eig_vec = eig_vec[:, idx]
    eig_val = eig_val[idx]
    print(eig_vec)
    numpc = 100  # THIS IS NUMBER OF PRINCIPAL COMPONENTS, YOU CAN CHANGE IT AND SEE RESULTS
    if numpc < p or numpc > 0:
        eig_vec = eig_vec[:, range(numpc)]
    score = np.dot(eig_vec.T, cov_mat)
    # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
    recon = np.dot(eig_vec, score) + np.mean(image_2d).T
    # TO CONTROL COMPLEX EIGENVALUES
    recon_img_mat = np.uint8(np.absolute(recon))
    return recon_img_mat


def guassian_2d(img):
    return cv2.GaussianBlur(img, (7, 7), 0)


def sobel_x(img):
    th = 200
    shape = img.shape
    print('sobelx{}'.format(shape))
    mean_x = np.mean(cv2.Sobel(img, cv2.CV_64F, 0, 1), axis=1)
    print(mean_x)
    img_new = []
    for i in mean_x:
        if i > th:
            img_new.append([0.99]*shape[1])
        else:
            img_new.append([0]*shape[1])
    img_new = np.array(img_new)
    print(img_new)
    img_new.astype(np.uint8)
    return img_new


def sobel_y(img):
    th = 0
    shape = img.shape
    print('sobely{}'.format(shape))
    mean_y = np.mean(cv2.Sobel(img, cv2.CV_64F, 0, 1), axis=0)
    print(mean_y)
    img_new = []
    for i in range(shape[0]):
        for j in mean_y:
            if j < -0.5:
                img_new.append(0.99)
            else:
                img_new.append(0)
    img_new = np.array(img_new).reshape(shape)
    print(img_new)
    img_new.astype(np.uint8)
    return img_new


def integral(img):
    return cv2.blur(img, (5, 5))


if __name__ == '__main__':
    img = cv2.imread('estate_original.jpg', 0)
    print(img)
    print('Begin filter!')
    pca_img = pca_2d(img)
    spe_img = specification_2d(img)
    sobelx_img = sobel_x(img)
    sobely_img = sobel_y(img)
    blur_img = guassian_2d(img)
    integral_img = integral(img)
    cv2.imshow('image', img)
    cv2.imshow('image_pca', pca_img)
    cv2.imshow('image_specification', spe_img)
    cv2.imshow('image_sobelx', sobelx_img)
    cv2.imshow('image_sobely', sobely_img)
    cv2.imshow('image_blur', blur_img)
    cv2.imshow('image_integral', integral_img)
    cv2.imwrite('./stock_graph/image.jpg', img)
    cv2.imwrite('./stock_graph/image_pca.jpg', pca_img)
    cv2.imwrite('./stock_graph/image_specification.jpg', spe_img)
    cv2.imwrite('./stock_graph/image_sobelx.jpg', sobelx_img)
    cv2.imwrite('./stock_graph/image_sobely.jpg', sobely_img)
    cv2.imwrite('./stock_graph/image_blur.jpg', blur_img)
    cv2.imwrite('./stock_graph/image_integral.jpg', integral_img)
    cv2.waitKey(200000)
    cv2.destroyAllWindows()