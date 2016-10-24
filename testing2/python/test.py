import cv2 as cv
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy

test_image = '../sample_image/nadal.png'
#test_image = '/home/lucie02/SkeletonDataset/no_consent/2016-10-12/2016-10-12_09:50:48_3d7959ac-03c5-5890-9771-e73e9789bc4f/rgb/rgb_00008.jpg'
img = np.float32(PIL.Image.open(test_image))
# imageToTest = cv.imread(test_image)
#
# cv.line(imageToTest,(340,350),(280,290),(255,0,0),5)
# cv.imshow('test',imageToTest)
# cv.waitKey(1000)
#
# imageToTest =  cv.cvtColor(imageToTest, cv.COLOR_BGR2RGB)
#img = img1[210:480,220:350,:]

start_time1 = time.time()

conf = 1
param, model = config_reader(conf)
boxsize = model['boxsize']
npart = model['np']
scale = boxsize/(img.shape[0] * 1.0)

imageToTest = cv.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

imageToTest_padded, pad = util.padRightDownCorner(imageToTest)
if param['use_gpu']:
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()
caffe.set_device(param['GPUdeviceNumber']) # set to your device!
person_net = caffe.Net(model['deployFile_person'], model['caffemodel_person'], caffe.TEST)
person_net.blobs['image'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
person_net.reshape()
person_net.forward(); # dry run to avoid GPU synchronization later in caffe

person_net.blobs['image'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
start_time = time.time()
output_blobs = person_net.forward()
print('Person net took %.2f ms.' % (1000 * (time.time() - start_time)))
person_map = np.squeeze(person_net.blobs[output_blobs.keys()[0]].data)

person_map_resized = cv.resize(person_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
data_max = scipy.ndimage.filters.maximum_filter(person_map_resized, 3)
maxima = (person_map_resized == data_max)
diff = (data_max > 0.5)
maxima[diff == 0] = 0
x = np.nonzero(maxima)[1]
y = np.nonzero(maxima)[0]

# person_map_to_plot_1 = util.colorize(person_map_resized) * 0.5 + imageToTest_padded * 0.5
# person_map_to_plot_2 = person_map_resized * 255
# for x_c, y_c in zip(x, y):
#     cv.circle(person_map_to_plot_2, (x_c, y_c), 3, (0,0,255), -1)
#util.showBGRimage('5-torso-overlayed',np.concatenate((person_map_to_plot_1, np.tile(person_map_to_plot_2[:,:,np.newaxis], (1,1,3))), axis=1))


num_people = x.size
person_image = np.ones((model['boxsize'], model['boxsize'], 3, num_people)) * 128
for p in range(num_people):
     for x_p in range(model['boxsize']):
         for y_p in range(model['boxsize']):
             x_i = x_p - model['boxsize']/2 + x[p]
             y_i = y_p - model['boxsize']/2 + y[p]
             if x_i >= 0 and x_i < imageToTest.shape[1] and y_i >= 0 and y_i < imageToTest.shape[0]:
                 person_image[y_p, x_p, :, p] = imageToTest[y_i, x_i, :]
# show one of them for inspection
# util.showBGRimage('6-center',person_image[:,:,:,0])

gaussian_map = np.zeros((model['boxsize'], model['boxsize']))
for x_p in range(model['boxsize']):
    for y_p in range(model['boxsize']):
        dist_sq = (x_p - model['boxsize']/2) * (x_p - model['boxsize']/2) + \
                  (y_p - model['boxsize']/2) * (y_p - model['boxsize']/2)
        exponent = dist_sq / 2.0 / model['sigma'] / model['sigma']
        gaussian_map[y_p, x_p] = math.exp(-exponent)
#util.showmap('7-center-torso',gaussian_map * 256)

pose_net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)
pose_net.forward() # dry run to avoid GPU synchronization later in caffe
output_blobs_array = [dict() for dummy in range(num_people)]
for p in range(num_people):
    input_4ch = np.ones((model['boxsize'], model['boxsize'], 4))
    input_4ch[:,:,0:3] = person_image[:,:,:,p]/256.0 - 0.5 # normalize to [-0.5, 0.5]
    input_4ch[:,:,3] = gaussian_map
    pose_net.blobs['data'].data[...] = np.transpose(np.float32(input_4ch[:,:,:,np.newaxis]), (3,2,0,1))
    start_time = time.time()
    if conf == 4:
         output_blobs_array[p] = copy.deepcopy(pose_net.forward()['Mconv5_stage4'])
    else:
         output_blobs_array[p] = copy.deepcopy(pose_net.forward()['Mconv7_stage6'])
    print('For person %d, pose net took %.2f ms.' % (p, 1000 * (time.time() - start_time)))

for p in range(num_people):
    print('Person %d' % p)
    #down_scaled_image = cv.resize(person_image[:,:,:,p], (0,0), fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    #canvas = np.empty(shape=(model['boxsize']/2, 0, 3))
    for part in [0,3,7,10,12]: # sample 5 body parts: [head, right elbow, left wrist, right ankle, left knee]
        part_map = output_blobs_array[p][0,part,:,:]
        part_map_resized = cv.resize(part_map, (0,0), fx=4, fy=4, interpolation=cv.INTER_CUBIC) #only for displaying
        #part_map_color = util.colorize(part_map_resized)
        #part_map_color_blend = part_map_color * 0.5 + down_scaled_image * 0.5
        #canvas = np.concatenate((canvas, part_map_color_blend), axis=1)
        #canvas = np.concatenate((canvas, 255 * np.ones((model['boxsize']/2, 5, 3))), axis=1)
    #util.showBGRimage('8-person-'+str(p)+'-pose',canvas)

prediction = np.zeros((14, 2, num_people))
for p in range(num_people):
    for part in range(14):
        part_map = output_blobs_array[p][0, part, :, :]
        part_map_resized = cv.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
        prediction[part,:,p] = np.unravel_index(part_map_resized.argmax(), part_map_resized.shape)
    # mapped back on full image
    prediction[:,0,p] = prediction[:,0,p] - (model['boxsize']/2) + y[p]
    prediction[:,1,p] = prediction[:,1,p] - (model['boxsize']/2) + x[p]

limbs = model['limbs']
stickwidth = 6
colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
[255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]] # note BGR ...
canvas = imageToTest.copy()
for p in range(num_people):
    for part in range(model['np']):
        cv.circle(canvas, (int(prediction[part, 1, p]), int(prediction[part, 0, p])), 3, (0, 0, 0), -1)
    for l in range(limbs.shape[0]):
        cur_canvas = canvas.copy()
        X = prediction[limbs[l,:]-1, 0, p]
        Y = prediction[limbs[l,:]-1, 1, p]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv.fillConvexPoly(cur_canvas, polygon, colors[l])
        canvas = canvas * 0.4 + cur_canvas * 0.6 # for transparency
	print int(mY),int(mX),int(length/2)

print('Overall time, pose net took %.2f ms.' % (1000 * (time.time() - start_time1)))
util.showBGRimage('9-final',canvas)


