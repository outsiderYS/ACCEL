import os
import random
import shutil


def getData(dirPath):
    subDirs = os.listdir(dirPath)
    destdir ='./testdl'  #这个文件夹需要提前建好
    for dir in subDirs:
        tempDir = dirPath+'\\'+dir+'\\'
        if not os.path.exists(destdir+'\\'+dir+'\\'):
            os.mkdir(destdir+'\\'+dir+'\\')
        fs = os.listdir(tempDir)
        random.shuffle(fs)
        le = int(len(fs)*0.8)  #这个可以修改划分比例
        for f in fs[le:]:
            shutil.move(tempDir+f, destdir+'\\'+dir+'\\')


getData("./number")
