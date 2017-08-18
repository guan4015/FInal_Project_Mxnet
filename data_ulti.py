

'''
This file read the original images with the boxes labels and transform them into the Recfile that
could be used by mxnet.
'''



import mxnet as mx
import numpy as np
import cv2
import random

# get iterator
def get_iterator(path, data_shape, label_width, batch_size, shuffle=False):
    iterator = mx.io.ImageRecordIter(path_imgrec=path,
                                    data_shape=data_shape,
                                    label_width=label_width,
                                    batch_size=batch_size,
                                    shuffle=shuffle)
    return iterator


# Obtain the YOLO boxes description from the original box description
# THis description contains the location of the box in the 7X7 grids and the center of the box as a
# ratio with respect to the size of the square that contains the center. .
def get_YOLO_xy(bxy, grid_size=(7,7), dscale=32, sizet=224):
    cx, cy = bxy
    assert cx<=1 and cy<=1, "All should be < 1, but get {}, and {}".format(cx,cy)

    j = int(np.floor(cx/(1.0/grid_size[0])))
    i = int(np.floor(cy/(1.0/grid_size[1])))
    xyolo = (cx * sizet - j * dscale) / dscale
    yyolo = (cy * sizet - i * dscale) / dscale
    return [i, j, xyolo, yyolo]


# Resize the image to fit into YOLO model and generate the 7X7X5 tensor.
def imgResizeBBoxTransform(img, bbox, sizet, grid_size=(7,7,5), dscale=32):

    himg, wimg = img.shape[:2]
    imgR = cv2.resize(img, dsize=(sizet, sizet))
    bboxyolo = np.zeros(grid_size)
    for eachbox in bbox:
        cx, cy, w, h = eachbox
        cxt = 1.0*cx/wimg
        cyt = 1.0*cy/himg
        wt = 1.0*w/wimg
        ht = 1.0*h/himg
        assert wt<1 and ht<1
        i, j, xyolo, yyolo = get_YOLO_xy([cxt, cyt], grid_size, dscale, sizet)
        print "one yolo box is {}".format((i, j, xyolo, yyolo, wt, ht))
        label_vec = np.asarray([1, xyolo, yyolo, wt, ht])
        bboxyolo[i, j, :] = label_vec
    return imgR, bboxyolo


# Convert raw images to rec files
# First we read the data and its corresponding box
# Second, we use the imageResizeBBoxTransform function to generate resized image and YOLO box
# Third, we bundle them and
def toRecFile(imgroot, imglist, annotation, sizet, grid_size, dscale, document_name):

    record = mx.recordio.MXIndexedRecordIO("./DATA_rec/{}.idx".format(document_name),
                                           "./DATA_rec/{}.rec".format(document_name), 'w')
    for i in range(len(imglist)):
        imgname = imglist[i]
        img = cv2.imread(imgroot+imgname+'.jpg')
        bbox = annotation[imgname]
        print "Now is processing img {}".format(imgname)
        imgR, bboxR = imgResizeBBoxTransform(img, bbox, sizet, grid_size, dscale)
        # preprocessing for the YOLO boxes. Flattening it and obtain a soft_max like output
        header = mx.recordio.IRHeader(flag=0, label=bboxR.flatten(), id=0, id2=0)
        # package the image and YOLO output and write them into the recfile
        s = mx.recordio.pack_img(header, imgR, quality=100, img_fmt='.jpg')
        record.write_idx(i, s)
    print "JPG to rec is Done"
    record.close()

if __name__ == "__main__":
    # transform jpg to rec file
    imgroot = "./DATA/"
    annotation = np.load("./DATA/annotation_list.npy")[()]
    # The following generate the .rec file with full data
    imglist = annotation.keys()
    sizet = 224
    toRecFile(imgroot, imglist, annotation, sizet, (7,7,5), 32, "cat_full_xiao")

    # The following generate the .rec file with trainning data set.
    imglist_train = annotation.keys()[0:448]
    toRecFile(imgroot, imglist_train, annotation, sizet, (7, 7, 5), 32, "cat_train_xiao")

    # The following generate the .rec file with the validation data set
    imglist_val = annotation.keys()[448:544]
    toRecFile(imgroot, imglist_val, annotation, sizet, (7, 7, 5), 32, "cat_val_xiao")

    # The following generate the.rec file with the small dataset
    # The number 15 means that we take 15 samples.
    imglist_small = random.sample(annotation.keys(),15)
    imglist_small
    toRecFile(imgroot, imglist_val, annotation, sizet, (7, 7, 5), 32, "cat_small_xiao")

