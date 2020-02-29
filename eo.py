import numpy as np
import pandas as pd
import random
import math,time,sys
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier


################################################################################################################3
def sigmoid(gamma):     #convert to probability
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def Vfunction(gamma):
	val = 1 + gamma*gamma
	val = math.sqrt(val)
	val = gamma/val
	return abs(val)


def fitness(particle):
	cols=np.flatnonzero(particle)
	val=1
	if np.shape(cols)[0]==0:
		return val	
	# clf = RandomForestClassifier(n_estimators=300)
	clf=KNeighborsClassifier(n_neighbors=5)
	# clf=MLPClassifier( alpha=0.01, max_iter=1000) #hidden_layer_sizes=(1000,500,100)
	#cross=4
	#test_size=(1/cross)
	#X_train, X_test, y_train, y_test = train_test_split(trainX, trainy,  stratify=trainy,test_size=test_size)
	train_data=trainX[:,cols]
	test_data=testX[:,cols]
	clf.fit(train_data,trainy)
	val=1-clf.score(test_data,testy)

	#in case of multi objective  []
	set_cnt=sum(particle)
	set_cnt=set_cnt/np.shape(particle)[0]
	val=omega*val+(1-omega)*set_cnt
	return val

def onecount(particle):
	cnt=0
	for i in particle:
		if i==1.0:
			cnt+=1
	return cnt


def allfit(population):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i])     
		#print(acc[i])
	return acc

def initialize(partCount,dim):
	population=np.zeros((partCount,dim))
	minn = 1
	maxx = math.floor(0.5*dim)
	if maxx<minn:
		maxx = minn + 1
	
	for i in range(partCount):
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

def avg_concentration(eqPool,poolSize,dimension):
	# simple average
	# print(np.shape(eqPool[0]))
	(r,) = np.shape(eqPool[0])
	avg = np.zeros(np.shape(eqPool[0]))
	for i in range(poolSize):
		x = np.array(eqPool[i])
		avg = avg + x
	
	#print(avg)
	avg = avg/poolSize
	#print(avg)

	#not actual average; but voting
	# for i in range(dimension):
	# 	if avg[i]>=0.5:
	# 		avg[i] = 1
	# 	else:
	# 		avg[i] = 0

	return avg

	#weighted avg (using Correlation/MI)



def signFunc(x): #signum function? or just sign ?
	if x<0:
		return -1
	return 1

def neighbor(particle,population):
	percent = 30
	percent /= 100
	numFeatures = np.shape(population)[1]
	numChange = int(numFeatures*percent)
	pos = np.random.randint(0,numFeatures-1,numChange)
	particle[pos] = 1 - particle[pos]
	return particle

def SA(population,accList):
	#dispPop()
	[partCount,numFeatures] = np.shape(population)
	T0 = numFeatures
	#print('T0: ',T0)
	for partNo in range(partCount):
		T=2*numFeatures
		curPar = population[partNo].copy()  
		curAcc = accList[partNo].copy()  
		#print('Par:',partNo, 'curAcc:',curAcc, 'curFeat:', onecount(curPar), 'fitness_check:', fitness(curPar))
		bestPar = curPar.copy()
		bestAcc = curAcc.copy()
		while T>T0:
			#print('T: ',T)
			newPar = neighbor(curPar,population)
			newAcc = fitness(newPar)/1.0
			if newAcc<bestAcc:
				curPar=newPar.copy()
				curAcc=newAcc.copy()
				bestPar=curPar.copy()
				bestAcc=curAcc.copy()
			elif newAcc==bestAcc:
				if onecount(newPar)<onecount(bestPar):
					curPar=newPar.copy()
					curAcc=newAcc
					bestPar=curPar.copy()
					bestAcc=curAcc
			else:            
				prob=np.exp((bestAcc-curAcc)/T)
				if(random.random()<=prob):
					curPar=newPar.copy()
					curAcc=newAcc
			T=int(T*0.7)
		#print('bestAcc: ',bestAcc)
		#print('Par:',partNo, 'newAcc:',bestAcc, 'newFeat:', onecount(bestPar), 'fitness_check: ', fitness(bestPar))
		population[partNo]=bestPar.copy() 
		accList[partNo]=bestAcc.copy()
	return population

def EO_SA(population,poolSize,max_iter,partCount,dimension):
	eqPool = np.zeros((poolSize+1,dimension))
	# print(eqPool)
	eqfit = np.zeros(poolSize+1)
	# print(eqfit)
	for i in range(poolSize+1):
		eqfit[i] = 100
	for curriter in range(max_iter):
	# print("iter no: ",curriter)
		# print(eqPool)
		popnew = np.zeros((partCount,dimension))
		accList = allfit(population)
		# x_axis.append(curriter)
		# y_axis.aend(min(accList))
		for i in range(partCount):
			for j in range(poolSize):
				if accList[i] <= eqfit[j]:
					eqfit[j] = accList[i].copy()
					eqPool[j] = population[i].copy()
					break
		
		# print("till best: ",eqfit[0],onecount(eqPool[0]))
		Cave = avg_concentration(eqPool,poolSize,dimension)
		eqPool[poolSize] = Cave.copy()

		t = (1 - (curriter/max_iter))**(a2*curriter/max_iter)
		for i in range(partCount):
				#randomly choose one candidate from the equillibrium pool
			random.seed(time.time() + 100 + 0.02*i)
			inx = random.randint(0,poolSize)
			Ceq = np.array(eqPool[inx])

			lambdaVec = np.zeros(np.shape(Ceq))
			rVec = np.zeros(np.shape(Ceq))
			for j in range(dimension):
				random.seed(time.time() + 1.1)
				lambdaVec[j] = random.random()
				random.seed(time.time() + 10.01)
				rVec[j] = random.random()
						
			FVec = np.zeros(np.shape(Ceq))
			for j in range(dimension):
				x = -1*lambdaVec[j]*t 
				x = math.exp(x) - 1
				x = a1 * signFunc(rVec[j] - 0.5) * x
	 				
			random.seed(time.time() + 200)
			r1 = random.random()
			random.seed(time.time() + 20)
			r2 = random.random()
			if r2 < GP:
				GCP = 0
			else:
				GCP = 0.5 * r1
			G0 = np.zeros(np.shape(Ceq))
			G = np.zeros(np.shape(Ceq))
			for j in range(dimension):
				G0[j] = GCP * (Ceq[j] - lambdaVec[j]*population[i][j])
				G[j] = G0[j]*FVec[j]
				# print('popnew[',i,']: ')
			for j in range(dimension):
				temp = Ceq[j] + (population[i][j] - Ceq[j])*FVec[j] + G[j]*(1 - FVec[j])/lambdaVec[j]
				temp = Vfunction(temp)
				if temp>0.5:
					popnew[i][j] = 1 - population[i][j]
				else:
					popnew[i][j] = population[i][j]
				# 	print(popnew[i][j],end=',')
				# print()

		population = popnew.copy()
		popnew = SA(popnew,accList)
		population = popnew.copy()	

	return eqPool,population

############################################################################################################
datasets=["BreastCancer","BreastEW","CongressEW","Exactly","Exactly2","HeartEW","Ionosphere","KrVsKpEW","Lymphography","M-of-n","PenglungEW","Sonar","SpectEW","Tic-tac-toe","Vote","WaveformEW","Wine","Zoo"]


for dataset in datasets:
	maxRun = 21
	print(dataset)
	if dataset == "KrVsKpEW" or dataset == "WaveformEW":
		maxRun = 2
		continue
	omega = 0.9 #weightage for no of features and accuracy
	partCountAll = [10]
	max_iterAll = [20]
	a2 = 1
	a1 = 2
	GP = 0.5
	poolSize = 4

	best_accuracy=np.zeros((1,4))
	best_no_features=np.zeros((1,4))
	#best_time_req=float(np.zeros((1,4)))
	
	best_accuracy = -1
	best_no_features = -1
	accuracy_list = []
							
	for runNo in range(maxRun):
		print(runNo)
		#===============================================================================================================
		df=pd.read_csv("Data/"+dataset+".csv")
		(a,b)=np.shape(df)
		# print(a,b)
		data = df.values[:,0:b-1]
		label = df.values[:,b-1]
		dimension = np.shape(data)[1] #particle dimension
		#===============================================================================================================

		cross = 5
		test_size = (1/cross)
		trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size)


		# clf = RandomForestClassifier(n_estimators=300)
		clf=KNeighborsClassifier(n_neighbors=5)
		# clf=MLPClassifier(alpha=0.001, max_iter=1000) #hidden_layer_sizes=(1000,500,100)
		clf.fit(trainX,trainy)
		val=clf.score(testX,testy)
		whole_accuracy = val
		print("Total Acc: ",val)

		for partCount in partCountAll:
			count=0
			for max_iter in max_iterAll:
				
				start_time = datetime.now()
				population = initialize(partCount,dimension)				
				[eqPool,population] = EO_SA(population,poolSize,max_iter,partCount,dimension)		
				# print(eqPool)
				time_required = datetime.now() - start_time

				# pyplot.plot(x_axis,y_axis)
				# pyplot.xlim(0,max_iter)
				# pyplot.ylim(max(0,min(y_axis)-0.1),min(max(y_axis)+0.1,1))
				# pyplot.show()


				output = eqPool[0].copy()
				# print(output)
				#test accuracy
				cols = np.flatnonzero(output)
				#print(cols)
				X_test = testX[:,cols]
				X_train = trainX[:,cols]
				#print(np.shape(feature))

				# clf = RandomForestClassifier(n_estimators=300)
				clf=KNeighborsClassifier(n_neighbors=5)
				#clf=MLPClassifier( alpha=0.001, max_iter=2000) #hidden_layer_sizes=(1000,500,100 ),
				clf.fit(X_train,trainy)
				val=clf.score(X_test, testy )
				accuracy_list.append(val)
				if val>best_accuracy:
					best_accuracy = val
					best_no_features = onecount(output)
				#average_accuracy += val
				# if ( val == best_accuracy[0,count] ) and ( onecount(output) < best_no_features[0,count] ):
				# 	best_accuracy[0,count] = val
				# 	best_no_features[0,count] = onecount( output )
				# 	#best_time_req[0,count] = time_required
				# 	best_whole_accuracy = whole_accuracy
					
				# if val > best_accuracy[0,count] :
				# 	best_accuracy[0,count] = val
				# 	best_no_features[0,count] = onecount( output )
				# 	#best_time_req[0,count] = time_required
				# 	best_whole_accuracy = whole_accuracy

				# print('best: ',best_accuracy[0,count], best_no_features[0,count])
				# print('avg: ',average_accuracy/10)
				# print("count:",count,"%.2f" % (100*best_accuracy[0,count]),best_no_features[0,count])
				count=count+1

				
	with open("list_EOvSA.csv","a") as f:
		print(dataset,file=f,end=',')
		for i in accuracy_list:
			print(i,file=f,end=',')
		print('',file=f)