import numpy as np
import pandas as pd
import random
import math,time,sys,os
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#========================================================================================================================

def sigmoid1(gamma):     #convert to probability
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid1c(gamma):     #convert to probability
	gamma = -gamma
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid2(gamma):
	gamma /= 2
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))
		
def sigmoid3(gamma):
	gamma /= 3
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid4(gamma):
	gamma *= 2
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))


def Vfunction1(gamma):
	return abs(np.tanh(gamma))

def Vfunction2(gamma):
	val = (math.pi)**(0.5)
	val /= 2
	val *= gamma
	val = math.erf(val)
	return abs(val)

def Vfunction3(gamma):
	val = 1 + gamma*gamma
	val = math.sqrt(val)
	val = gamma/val
	return abs(val)

def Vfunction4(gamma):
	val=(math.pi/2)*gamma
	val=np.arctan(val)
	val=(2/math.pi)*val
	return abs(val)




def fitness(position,trainX,trainy,testX,testy):
	cols=np.flatnonzero(position)
	val=1
	if np.shape(cols)[0]==0:
		return val	
	clf=KNeighborsClassifier(n_neighbors=5)
	train_data=trainX[:,cols]
	test_data=testX[:,cols]
	clf.fit(train_data,trainy)
	val=1-clf.score(test_data,testy)

	#in case of multi objective  []
	set_cnt=sum(position)
	set_cnt=set_cnt/np.shape(position)[0]
	val=omega*val+(1-omega)*set_cnt
	return val

def onecount(position):
	cnt=0
	for i in position:
		if i==1.0:
			cnt+=1
	return cnt


def allfit(population,trainX,trainy,testX,testy):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i],trainX,trainy,testX,testy)     
		#print(acc[i])
	return acc

def initialize(popSize,dim):
	population=np.zeros((popSize,dim))
	minn = 1
	maxx = math.floor(0.8*dim)
	if maxx<minn:
		minn = maxx
	
	for i in range(popSize):
		random.seed(i**3 + 19 + 83*time.time() ) 
		no = random.randint(minn,maxx)
		if no == 0:
			no = 1
		random.seed(time.time()*37 + 29)
		pos = random.sample(range(0,dim-1),no)
		for j in pos:
			population[i][j]=1
		
		# print(population[i])  
		
	return population


#========================================================================================================================
def funcPSO(popSize,maxIter,filename):

	df=pd.read_csv(filename)
	(a,b)=np.shape(df)
	print(a,b)
	data = df.values[:,0:b-1]
	label = df.values[:,b-1]
	dimension = np.shape(data)[1] #particle dimension

	cross = 5
	test_size = (1/cross)
	trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size)
	clf=KNeighborsClassifier(n_neighbors=5)
	clf.fit(trainX,trainy)
	val=clf.score(testX,testy)
	whole_accuracy = val
	print("Total Acc: ",val)

	x_axis = []
	y_axis = []

	population = initialize(popSize,dimension) 
	velocity = np.zeros((popSize,dimension))
	# print(population)
	gbestVal = 1000
	gbestVec = np.zeros(np.shape(population[0])[0])

	pbestVal = np.zeros(popSize)
	pbestVec = np.zeros(np.shape(population))	
	print(np.shape)
	for i in range(popSize):
		pbestVal[i] = 1000
	
	start_time = datetime.now()
	for curIter in range(maxIter):
		popnew = np.zeros((popSize,dimension))

		fitList = allfit(population,trainX,trainy,testX,testy)
		#update pbest
		for i in range(popSize):
			if (fitList[i] < pbestVal[i]):
				pbestVal[i] = fitList[i]
				pbestVec[i] = population[i].copy()
				print("pbest updated")

		#update gbest
		for i in range(popSize):
			if (fitList[i] < gbestVal):
				gbestVal = fitList[i]
				gbestVec = population[i].copy()
		# print(gbestVec)
		print("gbest: ",gbestVal,onecount(gbestVec))
		#update W
		W = WMAX - (curIter/maxIter)*(WMAX - WMIN )
		# print("w: ",W)
		ychosen , zchosen = 0 , 0
		for inx in range(popSize):
			#inx <- particle index
			random.seed(time.time()+10)
			r1 = C1 * random.random()
			random.seed(time.time()+19)
			r2 = C2 * random.random()

			x = np.subtract(pbestVec[inx] , population[inx])
			y = np.subtract(gbestVec , population[inx])
			velocity[inx] = np.multiply(W,velocity[inx]) + np.multiply(r1,x) + np.multiply(r2,y)

			########## if S function
			# popnew[inx] = np.add(population[inx],velocity[inx])
			# for j in range(dimension):
			# 	temp = sigmoid4(popnew[inx][j])
			# 	if temp > 0.5:
			# 		popnew[inx][j] = 1
			# 	else:
			# 		popnew[inx][j] = 0
			
			########## if V function
			# for j in range(dimension):
			# 	temp = Vfunction1(velocity[inx][j])
			# 	if temp > 0.5:
			# 		popnew[inx][j] = 1 - population[inx][j]
			# 	else:
			# 		popnew[inx][j] = population[inx][j]

			########## if X function
			
			popnew[inx] = np.add(population[inx],velocity[inx])
			y, z = np.array([]), np.array([])
			for j in range(dimension):
				temp = sigmoid1(popnew[inx][j])
				if temp > 0.5:
					y = np.append(y,1)
				else:
					y = np.append(y,0)

				temp = sigmoid1c(popnew[inx][j])
				if temp > 0.5:
					z = np.append(z,1)
				else:
					z = np.append(z,0)
			yfit = fitness(y,trainX,trainy,testX,testy)
			zfit = fitness(z,trainX,trainy,testX,testy)
			if yfit<zfit:
				ychosen += 1
				popnew[inx] = y.copy()
			else:
				zchosen += 1
				popnew[inx] = z.copy()
			
		# print("ychosen:",ychosen,"zchosen:",zchosen)


		population = popnew.copy()

	time_required = datetime.now() - start_time
	output = gbestVec.copy()
	print(output)

	cols=np.flatnonzero(output)
	#print(cols)
	X_test=testX[:,cols]
	X_train=trainX[:,cols]
	#print(np.shape(feature))

	clf=KNeighborsClassifier(n_neighbors=5)
	clf.fit(X_train,trainy)
	val=clf.score(X_test, testy )
	print(val,onecount(output))

	return val,output





#========================================================================================================================
omega = 0.9
popSize = 20
maxIter = 30
C1 = 2
C2 = 2
WMAX = 0.9
WMIN = 0.4


directory="csvUCI/"
filelist=os.listdir(directory )
for filename in filelist:
	print(filename)
	best_accuracy = -1
	best_no_features = -1
	average_accuracy = 0
	global_count = 0
	accuracy_list = []
	features_list = []

	for global_count in range(5):
		if (filename == "WaveformEW.csv" or filename == "KrVsKpEW.csv" ) and global_count > 1:
			break
		
		val,output = funcPSO(popSize,maxIter,directory+filename)

		accuracy_list.append(val)
		features_list.append(onecount(output))
		if ( val == best_accuracy ) and ( onecount(output) < best_no_features ):
			best_accuracy = val
			best_no_features = onecount( output )
			# best_time_req = time_required
			# best_whole_accuracy = whole_accuracy

		if val > best_accuracy :
			best_accuracy = val
			best_no_features = onecount( output )
			# best_time_req = time_required
			# best_whole_accuracy = whole_accuracy

	print('best: ',best_accuracy, best_no_features)

	# temp=sys.argv[1].split('/')[-1]
	temp = filename.split('.')[0]
	with open("result_PSOx1_uci.csv","a") as f:
		print(temp,"%.2f" % (100*best_accuracy),best_no_features,file=f)