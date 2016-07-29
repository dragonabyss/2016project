import pywt
import numpy as np
from medpy.io import load


def xdwtfun(data, wname):
    matSize = np.shape(data)
    testL, testH = pywt.dwt(data[:, 0, 0], wname)
    L = np.zeros((testL.shape[0], matSize[1], matSize[2]), dtype='float64')
    H = np.zeros((testH.shape[0], matSize[1], matSize[2]), dtype='float64')
    for i in range(0, matSize[1]):
        for j in range(0, matSize[2]):
            line = np.float64(data[:, i, j])
            cL, cH = pywt.dwt(line, wname)
            L[:, i, j] = cL
            H[:, i, j] = cH
    return L, H

def xidwtfun(L, H, wname):
    # Ls = np.shape(L)
    # Hs = np.shape(H)
    matSize = np.shape(L)
    testData = pywt.idwt(L[:, 0, 0], H[:, 0, 0], wname)
    data = np.zeros((testData.shape[0], matSize[1], matSize[2]), dtype='float64')
    for i in range(0, matSize[1]):
        for j in range(0, matSize[2]):
            L1 = L[:, i, j]
            H1 = H[:, i, j]
            line = pywt.idwt(L1, H1, wname)
            data[:, i, j] = line;
    return data


def ydwtfun(data, wname):
    matSize = np.shape(data)
    testL, testH = pywt.dwt(data[0, :, 0], wname)
    L = np.zeros((matSize[0], testL.shape[0], matSize[2]), dtype='float64')
    H = np.zeros((matSize[0], testH.shape[0], matSize[2]), dtype='float64')
    for i in range(0, matSize[0]):
        for j in range(0, matSize[2]):
            line = np.float64(data[i, :, j])
            cL, cH = pywt.dwt(line, wname)
            L[i, :, j] = cL
            H[i, :, j] = cH
    return L, H


def yidwtfun(L, H, wname):
    # Ls = np.shape(L)
    # Hs = np.shape(H)
    matSize = np.shape(L)
    testData = pywt.idwt(L[0, :, 0], H[0, :, 0], wname)
    data = np.zeros((matSize[0], testData.shape[0], matSize[2]), dtype='float64')
    for i in range(0, matSize[0]):
        for j in range(0, matSize[2]):
            L1 = L[i, :, j]
            H1 = H[i, :, j]
            line = pywt.idwt(L1, H1, wname)
            data[i, :, j] = line;
    return data


def zdwtfun(data, wname):
    matSize = np.shape(data)
    testL, testH = pywt.dwt(data[0, 0, :], wname)
    L = np.zeros((matSize[0], matSize[1], testL.shape[0]), dtype='float64')
    H = np.zeros((matSize[0], matSize[1], testH.shape[0]), dtype='float64')
    for i in range(0, matSize[0]):
        for j in range(0, matSize[1]):
            line = np.float64(data[i, j, :])
            cL, cH = pywt.dwt(line, wname)
            L[i, j, :] = cL
            H[i, j, :] = cH
    return L, H


def zidwtfun(L, H, wname):
    # Ls = np.shape(L)
    # Hs = np.shape(H)
    matSize = np.shape(L)
    testData = pywt.idwt(L[0, 0, :], H[0, 0, :], wname)
    data = np.zeros((matSize[0], matSize[1], testData.shape[0]), dtype="float64")
    for i in range(0, matSize[0]):
        for j in range(0, matSize[1]):
            L1 = L[i, j, :]
            H1 = H[i, j, :]
            line = pywt.idwt(L1, H1, wname)
            data[i, j, :] = line;
    return data


def dwt3fun(data, wname):
    L, H = xdwtfun(data, wname)
    LL, LH = ydwtfun(L, wname)
    HL, HH = ydwtfun(H, wname)
    LLL, LLH = zdwtfun(LL, wname)
    LHL, LHH = zdwtfun(LH, wname)
    HLL, HLH = zdwtfun(HL, wname)
    HHL, HHH = zdwtfun(HH, wname)
    return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH


def idwt3fun(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH, wname):
    HH = zidwtfun(HHL, HHH, wname)
    HL = zidwtfun(HLL, HLH, wname)
    LH = zidwtfun(LHL, LHH, wname)
    LL = zidwtfun(LLL, LLH, wname)
    H = yidwtfun(HL, HH, wname)
    L = yidwtfun(LL, LH, wname)
    data = xidwtfun(L, H, wname)
    return data


def dwt3funN(data, N, wname):
    for i in range(0, N):
        result = dwt3fun(data, wname)
        data = result[0]
    return result


image_data, image_header = load(r'avg152T1_RL_nifti.nii.gz')
print image_data.shape, image_data.dtype
# LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH= dwt3fun(image_data,r'db2')
# idwt_data = idwt3fun(LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH,r'db2')
LLL = dwt3funN(image_data, 3, r'db2')
print LLL[0].shape
