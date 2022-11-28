import os
import random
import shutil


def getData(dirPath):
    subDirs = os.listdir(dirPath)
    destdir ='./testdl'  #这个文件夹需要提前建
    file = open("testdl.txt", "w")
    for dir in subDirs:
        tempDir = dirPath+'\\'+dir+'\\'
        if not os.path.exists(destdir+'\\'+dir+'\\'):
            os.mkdir(destdir+'\\'+dir+'\\')
        fs = os.listdir(tempDir)
        temp = []
        for i in fs:
            if i.split('.')[-1] == "png":
                temp.append(i)
        fs = temp
        random.shuffle(fs)
        le = int(len(fs)*0.8)  #这个可以修改划分比例
        for f in fs[le:]:
            shutil.move(tempDir+f, destdir+'\\'+dir+'\\')
            file.write(tempDir+f+"\n")
    file.close()


getData("./number")
