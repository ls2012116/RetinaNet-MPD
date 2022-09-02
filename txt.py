import glob, os

# 数据集的位置
imgs_dir = '/home/delta/home/E/MPD-master/HRSC2016/Train/AllImages'
print(imgs_dir)

# 用作 test 的图片数据的比例
percentage_test = 10;

# 创建训练数据集和测试数据集：train.txt 和 test.txt
file_train = open('/home/delta/home/E/MPD-master/HRSC2016/Train/train1.txt', 'w')
file_test = open('/home/delta/home/E/MPD-master/HRSC2016/Train/test1.txt', 'w')
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(imgs_dir, "*.bmp")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test:
        counter = 1
        file_test.write(title + "\n")
    else:
        file_train.write(title + "\n")
        counter = counter + 1
