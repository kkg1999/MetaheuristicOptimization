import cv2
import sys, math, time, random, os
from datetime import datetime
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import  structural_similarity as ssim
from sewar.full_ref import vifp
import imageio


#===================================================================
def sigmoid(gamma):     #convert to probability
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma/5))
	else:
		return 1/(1 + math.exp(-gamma/5))

def Vfunction(gamma):
	# return abs(np.tanh(gamma))

	#another V-shaped function
	val=(math.pi/2)*gamma
	val=np.arctan(val)
	val=(2/math.pi)*val
	return abs(val)

def initialize(popSize,dimension):
	population = np.zeros((popSize,dimension))
	minn = 0
	maxx = 255
	for i in range(popSize):
		for j in range(dimension):
			random.seed( i + j + time.time() )
			population[i][j] = random.randint(minn,maxx)
		population[i].sort()
		population[i][0] = 0
		population[i][dimension-1] = 255
	return population


def fitness(agentCurr,inputAgent,inputImage,iterNo):
	print("fitness function executing")
	# laplacian = cv2.Laplacian(img,cv2.CV_64F)
	img = deepcopy(inputImage)
	img2 = transformImage(agentCurr,inputAgent,inputImage,iterNo)
	edges = cv2.Canny(img2,100,200)
	count=0
	intsum=1.1      
	tmp=deepcopy(img2)
	dimensions=img2.shape
	for r in range(0,dimensions[0]):
		for c in range(0,dimensions[1]):
			if (edges[r,c]==255):
				count+=1
				tmp[r,c]=-1
				intsum+=img2[r,c]
	#entropy of image
	flatImg =  [x for sublist in img2 for x in sublist]
	uniqImg = set(flatImg)
	Hx = 0
	for x in uniqImg:
		p_x = flatImg.count(x) / len(uniqImg)
		Hx += ((- x) * (math.log2(p_x)))
	#Well, How about calling the in built entropy function?
	#Yep'
	# will try that one if this disappoints me
	# print('sum = ',sum,end=' ')
	# print('log(sum) = ',math.log(sum))
	# tmp=deepcopy(img2)
	# stepSize = 5
	# localcontrast=1.1
	# (w_width, w_height) = (5, 5)
	# for row in range(0, tmp.shape[0] - w_height, stepSize):
	# 	for col in range(0, tmp.shape[1] - w_width , stepSize):
	# 		tmp2 = tmp[row:row + w_width, col:col + w_height]
	# 		localcontrast+=((max(tmp2.flatten()[~np.isin(tmp2.flatten(),-1)]))-(min(tmp2.flatten()[~np.isin(tmp2.flatten(),-1)])))
	# meanDiff=(1.02**(-(np.mean(img2)-np.mean(img))**2))
	# pixRange=((max(img2.flatten())-min(img2.flatten()))**2)
	# fit=math.log(math.log(intsum))*Hx*(count)*meanDiff*pixRange*localcontrast
	fit = math.log(math.log(intsum))*(count)
	# print(fit)
	return fit


def allfit(population,inputAgent,inputImage,iterNo):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i],inputAgent,inputImage,iterNo)     
		# print(acc[i])
	return acc


def transformImage(currAgent,inputAgent,inputImage,iterNo):
	tarnsImage = inputImage.copy()
	row = np.shape(inputImage)[0]
	col = np.shape(inputImage)[1]
	currAgent.sort()
	for i in range(row):
		for j in range(col):
			k = inputAgent.index(tarnsImage[i][j])
			tarnsImage[i][j] = currAgent[k]
	# cv2.imwrite("intermediate/"+str(iterNo)+'_'+str(agentNo)+'.png',tarnsImage)
	# cv2.imshow('new image',tarnsImage)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return tarnsImage


# In[5]:
#===================================================================================

def bmoIE(imageName,popSize,maxIter):
	img = cv2.imread(imageName,0)
	# img = imageio.imread(imageName,as_gray=True)
	print("image shape:",np.shape(img))
	#####################
	# contrast reduction
	alpha = 0.2 # contrast of input image = alpha * contrast of original image
	beta = ((1 - alpha )*img.sum())/(img.shape[0]*img.shape[1])
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			img[y,x] = int(np.clip(alpha*img[y,x] + beta, 0, 255))
	inputImageName = imageName.split('/')[-1]
	inputImageName = "i_"+inputImageName
	cv2.imwrite("input/"+inputImageName,img)
	# imageio.imsave("input/"+inputImageName,img)

	histr = cv2.calcHist([img],[0],None,[256],[0,256])
	plt.plot(histr)
	plt.savefig("histograms/hi_"+imageNameList[i]) #histogram of output image
	plt.clf()
	#####################
	inputImage = deepcopy(img)
	# print(np.shape(img))
	flatImg =  [x for sublist in img for x in sublist]
	# print(flatImg)
	uniqImg = set(flatImg)
	agentLength = len(uniqImg)
	print(agentLength)
	inputAgent = list(uniqImg)
	inputAgent.sort()
	inputImage = deepcopy(img)
	# print(inputAgent)

	population = initialize(popSize,agentLength)
	dimension = agentLength

	start_time = datetime.now()
	fitList = allfit(population,inputAgent,img,0)
	# temp = [-x for x in fitList]
	#sort agents
	# population = [x for _,x in sorted(zip(temp,population))]
	arr1inds = fitList.argsort()
	fitList = fitList[arr1inds[::-1]]
	population = population[arr1inds[::-1]]

	# fitList.sort()
	fitList = list(fitList)
	# fitList.reverse()

	
	receivedList = []
	for currIter in range(1,maxIter):
		area = int(0.5 * popSize)
		random.seed(time.time() + 10 )
		parent1 = random.randint(0,popSize-1)
		random.seed(time.time() + 19 )
		parent2 = random.randint(0,popSize-1)
		while(parent2 == parent1 and parent2 in receivedList):
			parent2 = random.randint(0,popSize-1)
		receivedList.append(parent2)
		# print(fitList[parent1],fitList[parent2])
		
		random.seed(time.time() + 29 )
		offspring = np.multiply(random.random(),population[parent2])
		if abs(parent1 - parent2 )<= area:
			p = random.uniform(0,1) # we can make this ratio based on their fitness
			q = (1 - p)
			offspring = np.add(np.multiply(p,population[parent1]),np.multiply(q,population[parent2]))

		currFit = fitness(offspring,inputAgent,inputImage,currIter)
		inx = 0
		while inx<popSize and fitList[inx]>currFit:
		  inx += 1
		if inx<popSize:
			population = np.insert(population,inx,offspring,axis=0)
			population = np.delete(population,popSize-1,axis=0)
			if inx in receivedList:
			  receivedList.remove(inx)
			fitList.insert(inx,currFit)
	bestAgent = population[0].copy()
	bestImage = transformImage(bestAgent,inputAgent,inputImage,maxIter)
	# transImageName = imageName.split('/')[-1]
	# transImageName = "o_"+transImageName
	# cv2.imwrite("output/"+transImageName,bestImage)
	time_req = datetime.now() - start_time
	print("time req:", time_req)

# 	histr = cv2.calcHist([bestImage],[0],None,[256],[0,256]) 
# 	plt.plot(histr)
# 	plt.show() 

	return bestImage,fitList[0]
#============================================================
popSize = 5
maxIter = 10
# imageNameList = sorted(os.listdir("GroundTruth/fivek/"))
#  'kodim02.png', 'kodim03.png', 'kodim04.png', 'kodim05.png']
# imageNameList = ['kodim01.png',     'kodim13.png',   'kodim25.png',
# 'kodim02.png',,  'kodim15.png',  'kodim18.png' ,  'kodim24.png'
# 'kodim01.png','kodim02.png','kodim03.png','kodim04.png','kodim05.png','kodim06.png','kodim07.png','kodim08.png','kodim09.png',
# 'kodim10.png','kodim11.png','kodim12.png','kodim13.png','kodim14.png','kodim15.png','kodim16.png','kodim17.png','kodim18.png',
# 'kodim19.png','kodim20.png','kodim21.png','kodim22.png','kodim23.png','kodim24.png'
# ,kodim25.png
imageNameList = ['kodim01.png','kodim02.png','kodim03.png','kodim04.png','kodim05.png','kodim06.png','kodim07.png','kodim08.png','kodim09.png','kodim10.png','kodim11.png','kodim12.png','kodim13.png','kodim14.png','kodim15.png','kodim16.png','kodim17.png','kodim18.png','kodim19.png','kodim20.png','kodim21.png','kodim22.png','kodim23.png','kodim24.png']

# imageNameList = [  'Budhha.png' , 'Fish.png' , 'Turbid.png']
# imageNameList = [ 'dibco16_1.png' , 'dibco16_2.png',  'dibco16_4.png',  'dibco16_6.png'  ,'dibco16_8.png' ,'dibco16_3.png' , 
# 'dibco16_5.png',  'dibco16_7.png',  'dibco16_9.png', 'dibco16_10.png']

print(imageNameList)

averagePsnr = 0
averageSsim = 0
averageVif = 0
for i in range(len(imageNameList)):
	print(imageNameList[i])
	####### inputName = "GroundTruth/kodakDataset/"+imageNameList[i]
	# truthName = imageNameList[i]
	# truthName = "DIBCO2016/"+imageNameList[i]
	truthName = "GroundTruth/kodakDataset/"+imageNameList[i]


	trImg = cv2.imread(truthName,0)
	# trImg = imageio.imread(truthName,as_gray=True)
	histr = cv2.calcHist([trImg],[0],None,[256],[0,256])
	plt.plot(histr)
	plt.savefig("histograms/ht_"+imageNameList[i]) #histogram of ground truth
	plt.clf()
	cv2.imwrite("truth/t_"+imageNameList[i],trImg)
	# imageio.imsave("truth/t_"+imageNameList[i],trImg)

	
	bestImage = deepcopy(trImg)

	maxPsnr = 0
	maxSsim = 0
	maxVif = 0
	maxfit = -1000
	for iteration in range(1):
		outputImage,fitval = bmoIE(truthName,popSize,maxIter)

		truthImage = cv2.imread(truthName,0)
		psnrval = cv2.PSNR(truthImage,outputImage)
		psnrval/=100
		ssimval = ssim(truthImage,outputImage)
		vifval = vifp(truthImage,outputImage)
		print(iteration,psnrval,ssimval,vifval,int((psnrval+ssimval+vifval)*100))
		
		total =  psnrval+ssimval+vifval
		if total>maxfit:
			maxfit = total
			maxPsnr = psnrval
			maxSsim = ssimval
			maxVif = vifval
			bestImage = deepcopy(outputImage)
	averagePsnr += maxPsnr
	averageSsim += maxSsim
	averageVif += maxVif
	print(i+1,averagePsnr,averageSsim,averageVif)

	histr = cv2.calcHist([bestImage],[0],None,[256],[0,256])
	plt.plot(histr)
	plt.savefig("histograms/ho_6"+imageNameList[i]) #histogram of output image
	plt.clf()
	transImageName = "o_6"+imageNameList[i]
	cv2.imwrite("output/"+transImageName,bestImage)
	# imageio.imsave("output/"+transImageName,bestImage)
	with open("resultBMO6.txt","a") as f:
		print(imageNameList[i],maxPsnr,maxSsim,maxVif,file=f)




averagePsnr /= len(imageNameList)
averageSsim /= len(imageNameList)
averageVif /= len(imageNameList)
print(averagePsnr)
print(averageSsim)
print(averageVif)

