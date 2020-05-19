import numpy as np
import pandas as pd
import random
import math,time,sys, os
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#==================================================================
def initialize(popSize,dim):
	population=np.zeros((popSize,dim))
	minn = 1
	maxx = math.floor(0.8*dim)
	if maxx<minn:
		minn = maxx
	
	for i in range(popSize):
		random.seed(i**3 + 10 + time.time() ) 
		no = random.randint(minn,maxx)
		if no == 0:
			no = 1
		random.seed(time.time()+ 100)
		pos = random.sample(range(0,dim-1),no)
		for j in pos:
			population[i][j]=1
		
		# print(population[i])  
	return population

def fitness(solution, trainX, trainy, testX,testy):
	cols=np.flatnonzero(solution)
	val=1
	if np.shape(cols)[0]==0:
		return val	
	clf=KNeighborsClassifier(n_neighbors=5)
	train_data=trainX[:,cols]
	test_data=testX[:,cols]
	clf.fit(train_data,trainy)
	error=1-clf.score(test_data,testy)

	#in case of multi objective  []
	featureRatio = (solution.sum()/np.shape(solution)[0])
	val=omega*error+(1-omega)*featureRatio
	# print(error,featureRatio,val)
	return val

def allfit(population, trainX,  trainy,testX, testy):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i],trainX,trainy,testX,testy)     
		#print(acc[i])
	return acc

def selectParentRoulette(popSize,fitnList):
	# maxx=max(fitnList)
	fitnList = np.array(fitnList)
	# fitnList = fitnList/maxx
	# minn = min(fitnList)
	fitnList = 1- fitnList/fitnList.sum()

	# print(fitnList)
	random.seed(time.time()+19)
	val = random.uniform(0,fitnList.sum())
	for i in range(popSize):
		if val <= fitnList[i]:
			return i
		val -= fitnList[i]
	return -1


def randomwalk(agent,agentFit):
	percent = 30
	percent /= 100
	neighbor = agent.copy()
	size = np.shape(agent)[0]
	upper = int(percent*size)
	if upper <= 1 or upper>size:
		upper = size
	x = random.randint(1,upper)
	pos = random.sample(range(0,size - 1),x)
	for i in pos:
		neighbor[i] = 1 - neighbor[i]
	return neighbor

def adaptiveBeta(agent,agentFit, trainX, trainy,testX,testy):
	bmin = 0.1 #parameter: (can be made 0.01)
	bmax = 1
	maxIter = 10 # parameter: (can be increased )
	maxIter = int(max(10,10*agentFit))

	
	for curr in range(maxIter):
		neighbor = agent.copy()
		size = np.shape(agent)[0]
		neighbor = randomwalk(neighbor,agentFit)

		beta = bmin + (curr / maxIter)*(bmax - bmin)
		for i in range(size):
			random.seed( time.time() + i )
			if random.random() <= beta:
				neighbor[i] = agent[i]
		neighFit = fitness(neighbor,trainX,trainy,testX,testy)
		if neighFit <= agentFit:
			agent = neighbor.copy()
			agentFit = neighFit
	return (agent,agentFit)

#============================================================================
def geneticAlgo(dataset,popSize,maxIter,randomstate):

	#--------------------------------------------------------------------
	df=pd.read_csv(dataset)
	(a,b)=np.shape(df)
	print(a,b)
	data = df.values[:,0:b-1]
	label = df.values[:,b-1]
	dimension = np.shape(data)[1] #solution dimension
	#---------------------------------------------------------------------

	cross = 5
	test_size = (1/cross)
	trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=randomstate) #
	print(np.shape(trainX),np.shape(trainy),np.shape(testX),np.shape(testy))

	clf=KNeighborsClassifier(n_neighbors=5)
	clf.fit(trainX,trainy)
	val=clf.score(testX,testy)
	whole_accuracy = val
	print("Total Acc: ",val)

	x_axis = []
	y_axis = []
	population = initialize(popSize,dimension)
	GBESTSOL = np.zeros(np.shape(population[0]))
	GBESTFIT = 1000

	start_time = datetime.now()
	
	for currIter in range(1,maxIter):
		newpop = np.zeros((popSize,dimension))
		# intermediate = np.zeros((popSize,dimension))
		fitList = allfit(population,trainX,trainy,testX,testy)
		arr1inds = fitList.argsort()
		population = population[arr1inds]
		fitList= fitList[arr1inds]
		# print(fitList)

		# for i in range(popSize):
		# 	print('here',i,fitList[i])
		# print('sum:',fitList.sum())
		# if currIter==1:
		# 	y_axis.append(min(fitList))
		# else:
		# 	y_axis.append(min(min(fitList),y_axis[len(y_axis)-1]))
		# x_axis.append(currIter)

		bestInx = np.argmin(fitList)
		fitBest = min(fitList)
		print(currIter,'best:',fitBest,population[bestInx].sum())
		# print(population[bestInx])
		if fitBest<GBESTFIT:
			GBESTSOL = population[bestInx].copy()
			GBESTFIT = fitBest

		for selectioncount in range(int(popSize/2)):
			parent1 =   selectParentRoulette(popSize,fitList)
			parent2 = parent1
			while parent2 == parent1:
				random.seed(time.time())
				# parent2 = random.randint(0,popSize-1)
				parent2 = selectParentRoulette(popSize,fitList)

				# print(parent2)
			# print('parents:',parent1,parent2)
			parent1 = population[parent1].copy()
			parent2 = population[parent2].copy()
			#cross over between parent1 and parent2
			child1 = parent1.copy()
			child2 = parent2.copy()
			for i in range(dimension):
				random.seed(time.time())
				if random.uniform(0,1)<crossoverprob:
					child1[i]=parent2[i]
					child2[i]=parent1[i]
			i = selectioncount
			j = int(i+(popSize/2))
			# print(i,j)
			newpop[i]=child1.copy()
			newpop[j]=child2.copy()

		#mutation
		mutationprob = muprobmin + (muprobmax - muprobmin)*(currIter/maxIter)
		for index in range(popSize):
			for i in range(dimension):
				random.seed(time.time()+dimension+popSize)
				if random.uniform(0,1)<mutationprob:
					newpop[index][i]= 1- newpop[index][i]
		# for i in range(popSize):
			# print('before:',newpop[i].sum(),fitList[i])
			# newpop[i],fitList[i] = adaptiveBeta(newpop[i],fitList[i],trainX,trainy,testX,testy)
			# newpop[i],fitList[i] = deepcopy(mutation(newpop[i],fitList[i],trainX,trainy,testX,testy))
			# print('after:',newpop[i].sum(),fitList[i])

		population = newpop.copy()
	# pyplot.plot(x_axis,y_axis)
	# pyplot.show()

	#test accuracy
	cols = np.flatnonzero(GBESTSOL)
	val = 1
	if np.shape(cols)[0]==0:
		return GBESTSOL
	clf = KNeighborsClassifier(n_neighbors=5)
	train_data = trainX[:,cols]
	test_data = testX[:,cols]
	clf.fit(train_data,trainy)
	val = clf.score(test_data,testy)
	return GBESTSOL,val


#========================================================================================================
popSize = 10
maxIter = 20
omega = 0.9
crossoverprob = 0.6
muprobmin = 0.01
muprobmax = 0.3
# datasetList = ["Breastcancer"]
datasetList = ["Breastcancer", "BreastEW", "CongressEW", "Exactly", "Exactly2", "HeartEW", "IonosphereEW", "KrvskpEW", "Lymphography", "M-of-n", "PenglungEW", "SonarEW", "SpectEW", "Tic-tac-toe", "Vote", "WaveformEW", "WineEW", "Zoo"]
# datasetList = ["Breastcancer"]
randomstateList=[15,5,15,26,12,7,10,8,37,19,35,2,49,26,1,25,47,12]

for datasetinx in range(len(datasetList)): #
	dataset=datasetList[datasetinx]
	best_accuracy = -100
	best_no_features = 100
	best_answer = []
	accuList = []
	featList = []
	for count in range(5):
		if (dataset == "WaveformEW" or dataset == "KrvskpEW") : #and count>2 :
			break
		print(count)
		answer,testAcc = geneticAlgo("csvUCI/"+dataset+".csv",popSize,maxIter,randomstateList[datasetinx])
		print(testAcc,answer.sum())
		accuList.append(testAcc)
		featList.append(answer.sum())
		if testAcc>=best_accuracy and answer.sum()<best_no_features:
			best_accuracy = testAcc
			best_no_features = answer.sum()
			best_answer = answer.copy()
		if testAcc>best_accuracy:
			best_accuracy = testAcc
			best_no_features = answer.sum()
			best_answer = answer.copy()

		
	
	print(dataset,"best:",best_accuracy,best_no_features)
	# inx = np.argmax(accuList)
	# best_accuracy = accuList[inx]
	# best_no_features = featList[inx]
	# print(dataset,"best:",accuList[inx],featList[inx])
	with open("result_GA.csv","a") as f:
		print(dataset,"%.2f"%(100*best_accuracy),best_no_features,sep=',',file=f)
	# with open("result_SMOXarrayA.csv","a") as f:
	# 	print(dataset,end=',',file=f)
	# 	for i in accuList:
	# 		print("%.2f"%(100*i),end=',',file=f)
	# 	print('',file=f)

	# with open("result_SMOXarrayF.csv","a") as f:
	# 	print(dataset,end=',',file=f)
	# 	for i in featList:
	# 		print(int(i),end=',',file=f)
	# 	print('',file=f)
