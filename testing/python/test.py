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


# init
conf = 1
param, model = config_reader(conf)
boxsize = model['boxsize']
npart = model['np']
if param['use_gpu']:
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()
caffe.set_device(param['GPUdeviceNumber']) # set to your device!
person_net = caffe.Net(model['deployFile_person'], model['caffemodel_person'], caffe.TEST)
pose_net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

# time stuff
starts = {}
ends = {}
starts['all'] = []
ends['all'] = []
for i in range(1,11):
	starts['block_'+str(i)] = []
	ends['block_'+str(i)] = []

# main loop
total_images = np.arange(8,60)
for f in total_images:
	starts['all'].append(time.time())

	# block 1
	starts['block_1'].append(time.time())
	#test_image = '../sample_image/nadal.png'
	if f<10:
		image_num = 'rgb_0000'+str(f)
	elif f<100:
		image_num = 'rgb_000'+str(f)
	elif f<1000:
		image_num = 'rgb_00'+str(f)
	elif f<10000:
		image_num = 'rgb_0'+str(f)
	test_image = '/home/omari/rgb/'+image_num+'.jpg'
	img = cv.imread(test_image)
	scale = boxsize/(img.shape[0] * 1.0)
	imageToTest = cv.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
	imageToTest_padded, pad = util.padRightDownCorner(imageToTest)
	ends['block_1'].append(time.time())

	# block 2
	starts['block_2'].append(time.time())
	person_net.blobs['image'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
	person_net.reshape()
	#person_net.forward(); # dry run to avoid GPU synchronization later in caffe
	ends['block_2'].append(time.time())

	# block 3
	starts['block_3'].append(time.time())
	person_net.blobs['image'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
	output_blobs = person_net.forward()
	person_map = np.squeeze(person_net.blobs[output_blobs.keys()[0]].data)
	ends['block_3'].append(time.time())

	# block 4
	starts['block_4'].append(time.time())
	person_map_resized = cv.resize(person_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
	data_max = scipy.ndimage.filters.maximum_filter(person_map_resized, 3)
	maxima = (person_map_resized == data_max)
	diff = (data_max > 0.5)
	maxima[diff == 0] = 0
	x = np.nonzero(maxima)[1]
	y = np.nonzero(maxima)[0]
	ends['block_4'].append(time.time())

	# block 5
	starts['block_5'].append(time.time())
	num_people = x.size
	person_image = np.ones((model['boxsize'], model['boxsize'], 3, num_people)) * 128
	for p in range(num_people):
		x_max = x[p]+model['boxsize']/2
		x_min = x[p]-model['boxsize']/2
		y_max = y[p]+model['boxsize']/2
		y_min = y[p]-model['boxsize']/2
		if x_min < 0:
			xn_min = x_min*-1
			x_min = 0
		else:
			xn_min = 0
		if x_max > imageToTest.shape[1]:
			xn_max = model['boxsize'] - (x_max-imageToTest.shape[1])
			x_max = imageToTest.shape[1]
		else:
			xn_max = model['boxsize']
		if y_min < 0:
			yn_min = y_min*-1
			y_min = 0
		else:
			yn_min = 0
		if y_max > imageToTest.shape[0]:
			yn_max = model['boxsize'] - (y_max-imageToTest.shape[0])
			y_max = imageToTest.shape[0]
		else:
			yn_max = model['boxsize']	
		person_image[yn_min:yn_max, xn_min:xn_max, :, p] = imageToTest[y_min:y_max, x_min:x_max, :]
	ends['block_5'].append(time.time())

	# block 6
	starts['block_6'].append(time.time())
	gaussian_map = np.zeros((model['boxsize'], model['boxsize']))
	x_p = np.arange(model['boxsize'])
	y_p = np.arange(model['boxsize'])
	part1 = [(x_p - model['boxsize']/2) * (x_p - model['boxsize']/2), np.ones(model['boxsize'])]
	part2 = [np.ones(model['boxsize']), (y_p - model['boxsize']/2) * (y_p - model['boxsize']/2)]
	dist_sq = np.transpose(np.matrix(part1))*np.matrix(part2)
	exponent = dist_sq / 2.0 / model['sigma'] / model['sigma']
	gaussian_map = np.exp(-exponent)
	ends['block_6'].append(time.time())

	# block 7
	starts['block_7'].append(time.time())
	#pose_net.forward() # dry run to avoid GPU synchronization later in caffe
	output_blobs_array = [dict() for dummy in range(num_people)]
	for p in range(num_people):
	    input_4ch = np.ones((model['boxsize'], model['boxsize'], 4))
	    input_4ch[:,:,0:3] = person_image[:,:,:,p]/256.0 - 0.5 # normalize to [-0.5, 0.5]
	    input_4ch[:,:,3] = gaussian_map
	    pose_net.blobs['data'].data[...] = np.transpose(np.float32(input_4ch[:,:,:,np.newaxis]), (3,2,0,1))
	    if conf == 4:
		 output_blobs_array[p] = copy.deepcopy(pose_net.forward()['Mconv5_stage4'])
	    else:
		 output_blobs_array[p] = copy.deepcopy(pose_net.forward()['Mconv7_stage6'])
	ends['block_7'].append(time.time())

	# block 8
	starts['block_8'].append(time.time())
	for p in range(num_people):
	    print('Person %d' % p)
	    for part in [0,3,7,10,12]: # sample 5 body parts: [head, right elbow, left wrist, right ankle, left knee]
		part_map = output_blobs_array[p][0,part,:,:]
		part_map_resized = cv.resize(part_map, (0,0), fx=4, fy=4, interpolation=cv.INTER_CUBIC) #only for displaying
	ends['block_8'].append(time.time())

	# block 9
	starts['block_9'].append(time.time())
	prediction = np.zeros((14, 2, num_people))
	for p in range(num_people):
	    for part in range(14):
		part_map = output_blobs_array[p][0, part, :, :]
		part_map_resized = cv.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
		prediction[part,:,p] = np.unravel_index(part_map_resized.argmax(), part_map_resized.shape)
	    # mapped back on full image
	    prediction[:,0,p] = prediction[:,0,p] - (model['boxsize']/2) + y[p]
	    prediction[:,1,p] = prediction[:,1,p] - (model['boxsize']/2) + x[p]
	ends['block_9'].append(time.time())

	# block 10
	starts['block_10'].append(time.time())
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
	ends['block_10'].append(time.time())

	util.showBGRimage(image_num+'-final',canvas)
	ends['all'].append(time.time())

for i,j in zip(starts['all'],ends['all']):
	print 'the total time is:',j-i

for i in range(1,11):
	sum = 0.0
	for j in range(len(total_images)):
		sum+=ends['block_'+str(i)][j]-starts['block_'+str(i)][j]	
	print 'the avg of block_',i,' is:',sum/len(total_images)

