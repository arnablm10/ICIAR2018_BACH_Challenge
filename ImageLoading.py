Train_loc = 'drive/MyDrive/ICIAR2018_BACH_Challenge/Photos/'

ext = ['tif']    # Add image formats here
subdirectory = ['Benign', 'Normal', 'InSitu', 'Invasive']
files = []
[files.extend(glob.glob(Train_loc + sd + '/*.' + e)) for sd in subdirectory for e in ext]
# print(files)
X = [cv2.imread(file) for file in files]
X = np.asarray(X)

patch_dim = 512
Image_height = 1536
Image_width = 2048
stride = 256
patch_count = ((Image_height - patch_dim + 1 + stride)  // stride) * ((Image_width - patch_dim + 1 + stride) // stride)
