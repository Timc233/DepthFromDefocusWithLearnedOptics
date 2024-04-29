import imageio
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imsave
import os
import glob
import tqdm
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from skimage.color import rgb2gray

def fill_depth_colorization(imgRgb, imgDepth, alpha=1):
    if imgDepth.dtype != float:
        imgDepth = imgDepth.astype(float)

    imgIsNoise = (imgDepth == 0) | (imgDepth == 10)
    maxImgAbsDepth = np.max(imgDepth[~imgIsNoise])
    imgDepth = imgDepth / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1

    assert imgDepth.ndim == 2
    H, W = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape(H, W)

    knownValMask = ~imgIsNoise

    grayImg = rgb2gray(imgRgb)
    
    winRad = 1

    len_indices = 0
    cols = np.zeros(numPix * (2*winRad + 1)**2, dtype=int)
    rows = np.zeros(numPix * (2*winRad + 1)**2, dtype=int)
    vals = np.zeros(numPix * (2*winRad + 1)**2)
    gvals = np.zeros((2*winRad + 1)**2)

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i-winRad), min(i+winRad + 1, H)):
                for jj in range(max(0, j-winRad), min(j+winRad + 1, W)):
                    if ii == i and jj == j:
                        continue
                    rows[len_indices] = i * W + j
                    cols[len_indices] = ii * W + jj
                    gvals[nWin] = grayImg[ii, jj]
                    nWin += 1
                    len_indices += 1
            
            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1]))**2)

            csig = c_var * 0.6
            mgv = min((gvals[:nWin] - curVal)**2)
            if csig < (-mgv / np.log(0.01)):
                csig = -mgv / np.log(0.01)
            
            if csig < 0.000002:
                csig = 0.000002
            
            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal)**2 / csig)
            gvals[:nWin] /= np.sum(gvals[:nWin])
            vals[len_indices - nWin:len_indices] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_indices] = i * W + j
            cols[len_indices] = i * W + j
            vals[len_indices] = 1
            len_indices += 1

    vals = vals[:len_indices]
    cols = cols[:len_indices]
    rows = rows[:len_indices]
    A = csr_matrix((vals, (rows, cols)), shape=(numPix, numPix))
    
    G = diags(knownValMask.ravel() * alpha, 0, shape=(numPix, numPix))
    
    b = knownValMask.ravel() * imgDepth.ravel() * alpha
    new_vals = spsolve(A + G, b)
    new_vals = new_vals.reshape(H, W)
    
    denoisedDepthImg = new_vals * maxImgAbsDepth
    return denoisedDepthImg


DATASET_PATH = 'data/training_data/dualpixel'

TEST_PATH = os.path.join(DATASET_PATH, 'test')
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
ALPHA = 0.8

folders = [TEST_PATH, TRAIN_PATH]

for folder in folders:
    depth_path = os.path.join(folder, "merged_depth")
    rgb_path = os.path.join(folder, "scaled_images")
    inpaint_path = os.path.join(folder, "inpainted_depth")
    if not os.path.exists(inpaint_path):
        os.makedirs(inpaint_path)

    print(f"Processing {folder} folder")

    for subfolder in tqdm.tqdm(os.listdir(depth_path)):
        depth_image_path = glob.glob(os.path.join(depth_path, subfolder, '*_center.png'))[0]
        rgb_image_path = glob.glob(os.path.join(rgb_path, subfolder, '*_center.jpg'))[0]
        imgRgb = imageio.imread(rgb_image_path) / 255.0
        imgDepth = imageio.imread(depth_image_path)
        denoisedDepthImg = fill_depth_colorization(imgRgb, imgDepth, alpha=ALPHA)
        inpaint_subfolder_path = os.path.join(inpaint_path, subfolder)
        if not os.path.exists(inpaint_subfolder_path):
            os.makedirs(inpaint_subfolder_path)
            # Normalize the image data to [0, 1]
        min_val = np.min(denoisedDepthImg)
        max_val = np.max(denoisedDepthImg)
        denoisedDepthImg_normalized = (denoisedDepthImg - min_val) / (max_val - min_val)

        # Scale to full 8-bit range [0, 255]
        denoisedDepthImg_scaled = (denoisedDepthImg_normalized * 255).astype(np.uint8)
        imsave(os.path.join(inpaint_subfolder_path, 'inpaint_depth_center.png'), denoisedDepthImg_scaled)





