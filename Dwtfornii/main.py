#!/usr/bin/python
# -*- coding: utf-8 -*-
import pywt
import numpy as np
import os
from medpy.io import load
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.cross_validation import cross_val_score


stdPath = r"/mnt/hgfs/linux_shared/20160718/std"

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

def dataPreprocess(data,target):
    #normalize
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    data_normalized = scaler.transform(data)
    #PCA
    pca = decomposition.PCA(n_components =0.95)
    data_prime = pca.fit_transform(data_normalized)
    #binarize
    label_binarizer = preprocessing.LabelBinarizer()
    new_target = label_binarizer.fit_transform(target)
    return data_prime,new_target

def calDWTforClass(filepath,tag,level,wname):
    classPath = os.path.join(filepath,tag)
    for childfold in os.listdir(classPath):
        child_fold_path = os.path.join(classPath,childfold)
        if os.path.isfile(child_fold_path):
            continue
        for file in os.listdir(child_fold_path):
            if file.endswith(".nii.gz"):
                file_path = os.path.join(child_fold_path,file)
                image_data,image_header = load(file_path)
                coef = dwt3funN(image_data,level,wname)
                coef = coef[0].flatten()
                print coef.shape
                coef_file = os.path.join(child_fold_path,"coef_"+childfold+"_"+ wname+"_"+str(level)+".txt")
                np.savetxt(coef_file,coef,fmt="%.8e")

def createDataMatrixForEach(filepath,tag,wname,level):
    flag = True
    classPath = os.path.join(filepath, tag)
    for childfold in os.listdir(classPath):
        child_fold_path = os.path.join(classPath, childfold)
        if os.path.isfile(child_fold_path):
            continue
        for file in os.listdir(child_fold_path):
            if file.startswith("coef_") and file.__contains__(wname) and file.__contains__(str(level)+".txt"):
                coef_path = os.path.join(child_fold_path,file)
                if flag == True:
                    flag = False
                    data = np.loadtxt(coef_path)
                else:
                    another = np.loadtxt(coef_path)
                    data = np.vstack((data,another))
    data_matrix_name = "coef_"+tag+"_"+wname+"_"+str(level)+"_matrix.txt"
    data_matrix_path = os.path.join(classPath,data_matrix_name)
    np.savetxt(data_matrix_path,data)
    return data

def createDataMatrix(filePath,wname,level):
    flag = True
    # for subdir in os.listdir(filePath):
    #     createDataMatrixForEach(filePath,subdir)
    coef_list = []
    for subdir in os.listdir(filePath):
        sub_path = os.path.join(filePath,subdir)
        if os.path.isfile(sub_path):
            continue
        for file in os.listdir(sub_path):
            if file.startswith("coef_") and file.__contains__(wname) and file.__contains__("_"+ str(level)+"_"):
                file_path = os.path.join(sub_path,file)
                coef_list.append(file_path)
    target_size = []
    for file in coef_list:
        if flag == True:
            flag = False
            data = np.loadtxt(file)
            target_size.append(data.shape[0])
        else:
            another = np.loadtxt(file)
            target_size.append(another.shape[0])
            data = np.vstack((data,another))
    targer_1 = [1] * target_size[0]
    print target_size[0]
    target_2 = [2] * target_size[1]
    print target_size[1]
    target_3 = [3] * target_size[2]
    print target_size[2]
    target = np.array(targer_1+target_2+target_3)
    target_path = os.path.join(stdPath,"coef_target_"+wname+"_"+str(level)+"_matrix.txt")
    np.savetxt(target_path,target)
    data_path = os.path.join(stdPath,"coef_data_"+wname+"_"+str(level)+"_matrix.txt")
    np.savetxt(data_path,data)
    return data,target

def loadDataMatrix(filePath,wname,level):
    for file in os.listdir(filePath):
        if file.startswith("coef_") and file.__contains__(wname) and file.__contains__("_" +str(level)+"_"):
            file_path = os.path.join(filePath,file)
            if file.__contains__("data"):
                data = np.loadtxt(file_path)
            elif file.__contains__("target"):
                target = np.loadtxt(file_path)
    return data,target

def twoClassSVMClassifier(class1_data,class1_target,class2_data,class2_target):
    data_set = np.vstack((class1_data,class2_data))
    target_set = np.hstack((class1_target,class2_target))
    data_prime, new_target = dataPreprocess(data_set, target_set)
    base_svm = SVC(kernel='rbf',gamma=0.3,random_state=56)
    scores = cross_val_score(base_svm,data_prime,target_set,cv=5,scoring='accuracy')
    print scores
    # X_train, X_test, Y_train, Y_test = train_test_split(data_prime, target_set, test_size=0.2, random_state=36)
    # base_svm.fit(X_train, Y_train)
    # X_predict = base_svm.predict(X_test)
    # print np.mean(X_predict == Y_test)


def mainProcess(wname,level):
    # createDataMatrix(stdPath)
    data, target = loadDataMatrix(stdPath,wname,level)
    AD_data = data[0:26,:]
    AD_target = target[0:26]
    MCI_data = data[26:57,:]
    MCI_target = target[26:57]
    NC_data = data[57:91,:]
    NC_target = target[57:91]
    # twoClassSVMClassifier(MCI_data, MCI_target, NC_data, NC_target)
    twoClassSVMClassifier(AD_data,AD_target,NC_data,NC_target)
    # twoClassSVMClassifier(MCI_data, MCI_target,AD_data, AD_target)
    # data_prime, new_target = dataPreprocess(data, target)
    # X_train,X_test,Y_train,Y_test = train_test_split(data_prime,target,test_size=0.2,random_state=42)
    # base_svm = SVC()
    # base_svm.fit(X_train,Y_train)
    # X_predict = base_svm.predict(X_test)

wname = 'db2'
level = 3
# calDWTforClass(stdPath,"AD",level,wname)
# calDWTforClass(stdPath,"MCI",level,wname)
# calDWTforClass(stdPath,"NC",level,wname)
# createDataMatrixForEach(stdPath,"AD",wname,level)
# createDataMatrixForEach(stdPath,"MCI",wname,level)
# createDataMatrixForEach(stdPath,"NC",wname,level)
# createDataMatrix(stdPath,wname,level)
# mainProcess(wname,level)
# data,target = loadDataMatrix(stdPath)
# print "data process start"
# data_prime,new_target=dataPreprocess(data,target)
# print data_prime.shape
# print new_target
# print "data process finish"
mainProcess(wname,level)

# image_data, image_header = load(r'avg152T1_RL_nifti.nii.gz')
# print image_data.shape, image_data.dtype
# # LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH= dwt3fun(image_data,r'db2')
# # idwt_data = idwt3fun(LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH,r'db2')
# LLL = dwt3funN(image_data, 3, r'db2')
# print LLL[0].shape
