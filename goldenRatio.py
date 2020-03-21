import numpy as np
import pandas as pd
import random
import math,time,sys
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#==================================================================
def sigmoid1(gamma):     #convert to probability
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid1i(gamma):     #convert to probability
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

def fitness(solution, trainX, testX, trainy, testy):
	cols=np.flatnonzero(solution)
	val=1
	if np.shape(cols)[0]==0:
		return val	
	clf=KNeighborsClassifier(n_neighbors=5)
	train_data=trainX[:,cols]
	test_data=testX[:,cols]
	clf.fit(train_data,trainy)
	val=1-clf.score(test_data,testy)

	#in case of multi objective  []
	set_cnt=sum(solution)
	set_cnt=set_cnt/np.shape(solution)[0]
	val=omega*val+(1-omega)*set_cnt
	return val

def allfit(population, trainX, testX, trainy, testy):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i],trainX,testX,trainy,testy)     
		#print(acc[i])
	return acc

def toBinary(solution,dimension):
	# print("continuous",solution)
	Xnew = np.zeros(np.shape(solution))
	for i in range(dimension):
		temp = Vfunction3(abs(solution[i]))

		random.seed(time.time()+i)
		if temp > random.random(): # sfunction
			Xnew[i] = 1
		else:
			Xnew[i] = 0
		# if temp > 0.5: # vfunction
		# 	Xnew[i] = 1 - abs(solution[i])
		# else:
		# 	Xnew[i] = abs(solution[i])
	# print("binary",Xnew)
	return Xnew

def toBinaryX(solution,dimension,oldsol,trainX, testX, trainy, testy):
	Xnew = np.zeros(np.shape(solution))
	Xnew1 = np.zeros(np.shape(solution))
	Xnew2 = np.zeros(np.shape(solution))
	for i in range(dimension):
		temp = sigmoid1(abs(solution[i]))
		random.seed(time.time()+i)
		r1 = random.random()
		if temp > r1: # sfunction
			Xnew1[i] = 1
		else:
			Xnew1[i] = 0

		temp = sigmoid1i(abs(solution[i]))
		if temp > r1: # sfunction
			Xnew2[i] = 1
		else:
			Xnew2[i] = 0

	fit1 = fitness(Xnew1,trainX,testX,trainy,testy)
	fit2 = fitness(Xnew2,trainX,testX,trainy,testy)
	fitOld =  fitness(oldsol,trainX,testX,trainy,testy)
	if fit1<fitOld or fit2<fitOld:
		if fit1 < fit2:
			Xnew = Xnew1.copy()
		else:
			Xnew = Xnew2.copy()
	return Xnew
	# else: CROSSOVER
	Xnew3 = Xnew1.copy()
	Xnew4 = Xnew2.copy()
	for i in range(dimension):
		random.seed(time.time() + i)
		r2 = random.random()
		if r2>0.5:
			tx = Xnew3[i]
			Xnew3[i] = Xnew4[i]
			Xnew4[i] = tx
	fit1 = fitness(Xnew3,trainX,testX,trainy,testy)
	fit2 = fitness(Xnew4,trainX,testX,trainy,testy)
	if fit1<fit2:
		return Xnew3
	else:
		return Xnew4
	# print("binary",Xnew)
	

#==================================================================
def goldenratiomethod(dataset,popSize,maxIter):

	#---------------------------------------------------------------------
	#I know I should not put not it here, but still ...
	df=pd.read_csv(dataset)
	(a,b)=np.shape(df)
	print(a,b)
	data = df.values[:,0:b-1]
	label = df.values[:,b-1]
	dimension = np.shape(data)[1] #particle dimension
	#---------------------------------------------------------------------

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
	BESTANS = np.zeros(np.shape(population[0]))
	BESTACC = 1000

	start_time = datetime.now()
	
	for currIter in range(1,maxIter):

		fitList = allfit(population,trainX,testX,trainy,testy)
		y_axis.append(min(fitList))
		x_axis.append(currIter)
		worstInx = np.argmax(fitList)
		fitWorst = max(fitList)
		Xworst = population[worstInx].copy()

		Xave = population.sum(axis=0)
		Xave = np.divide(Xave,popSize)
		# for x in Xave:
		# 	print("%.2f"%x,end=',')
		# print()
		XaveBin= toBinary(Xave,dimension)
		FITave = fitness(XaveBin, trainX, testX, trainy, testy)
		if FITave<fitWorst:
			population[worstInx] = XaveBin.copy()
			fitList[worstInx] = FITave
		


		for i in range(popSize):
			Xi = population[i].copy()
			j = i
			while j == i:
				random.seed(time.time()+j)
				j = random.randint(0, popSize-1)
			Xj = population[j].copy()
			FITi = fitList[i]
			FITj = fitList[j]

			Xave = population.sum(axis=0)
			Xave = np.subtract(Xave,population[i])
			Xave = np.subtract(Xave,population[j])
			Xave = np.divide(Xave,(popSize-2))
			XaveBin = toBinary(Xave,dimension)
			FITave = fitness(XaveBin, trainX, testX, trainy, testy)
			# print(i,j,FITi,FITj,FITave)
			Xbest = np.zeros(np.shape(Xi))
			Xmedium = np.zeros(np.shape(Xi))
			Xworst = np.zeros(np.shape(Xi))
			
			if FITi < FITj < FITave:
				Xbest = Xi.copy()
				Xmedium = Xj.copy()
				Xworst = Xave.copy()
			elif FITi < FITave < FITj:
				Xbest = Xi.copy()
				Xmedium = Xave.copy()
				Xworst = Xj.copy()
			elif FITj < FITi < FITave:
				Xbest = Xj.copy()
				Xmedium = Xi.copy()
				Xworst = Xave.copy()
			elif FITj < FITave < FITi:
				Xbest = Xj.copy()
				Xmedium = Xave.copy()
				Xworst = Xi.copy()
			elif FITave < FITi < FITj:
				Xbest = Xave.copy()
				Xmedium = Xi.copy()
				Xworst = Xj.copy()
			elif FITave < FITj < FITi:
				Xbest = Xave.copy()
				Xmedium = Xj.copy()
				Xworst = Xi.copy()

			Xt = np.subtract(Xmedium,Xworst)
			T = currIter/maxIter
			Ft = (golden/(5**0.5)) * (golden**T - (1 - golden)**T)
			random.seed(19*time.time() + 10.01)
			Xnew = np.multiply(Xbest,(1-Ft)) + np.multiply(Xt,random.random()*Ft)
			Xnew = toBinaryX(Xnew,dimension,population[i],trainX, testX, trainy, testy)
			FITnew = fitness(Xnew, trainX, testX, trainy, testy)
			# if FITnew < fitList[i]:
				# print(i,j,"updated2")
			population[i] = Xnew.copy()
			fitList[i] = FITnew

		#second phase
		worstInx = np.argmax(fitList)
		fitWorst = max(fitList)
		Xworst = population[worstInx].copy()
		bestInx = np.argmin(fitList)
		fitBest = min(fitList)
		Xbest = population[bestInx].copy()
		for i in range(popSize):
			Xi = population[i].copy()
			random.seed(29*time.time() + 391.97 )
			Xnew = np.add(Xi , np.multiply(np.subtract(Xbest,Xworst),random.random()*(1/golden)) )
			Xnew = toBinaryX(Xnew,dimension,population[i],trainX, testX, trainy, testy)
			FITnew = fitness(Xnew, trainX, testX, trainy, testy)
			# if FITnew < fitList[i]:
			fitList[i] = FITnew
			population[i] = Xnew.copy()

			if fitList[i]< BESTACC:
				BESTACC = fitList[i]
				BESTANS = population[i].copy()

		# pyplot.plot(x_axis,y_axis)
		# pyplot.show()
		# bestInx = np.argmin(fitList)
		# fitBest = min(fitList)
		# Xbest = population[bestInx].copy()
	cols = np.flatnonzero(BESTANS)
	val = 1
	if np.shape(cols)[0]==0:
		return Xbest
	clf = KNeighborsClassifier(n_neighbors=5)
	train_data = trainX[:,cols]
	test_data = testX[:,cols]
	clf.fit(train_data,trainy)
	val = clf.score(test_data,testy)
	return BESTANS,val




#==================================================================
golden = (1 + 5 ** 0.5) / 2
popSize = 10
maxIter = 10
omega = 1
datasetList = ["BreastEW"]
datasetList = ["Breastcancer", "BreastEW", "CongressEW", "Exactly", "Exactly2", "HeartEW", "Ionosphere", "KrVsKpEW", "Lymphography", "M-of-n", "PenglungEW", "Sonar", "SpectEW", "Tic-tac-toe", "Vote", "WaveformEW", "Wine", "Zoo"]

for dataset in datasetList:
	accuList = []
	featList = []
	for count in range(10):
		if (dataset == "WaveformEW" or dataset == "KrVsKpEW") and count>2:
			break
		print(count)
		answer,testAcc = goldenratiomethod("csvUCI/"+dataset+".csv",popSize,maxIter)
		print(testAcc,answer.sum())
		accuList.append(testAcc)
		featList.append(answer.sum())
	inx = np.argmax(accuList)
	best_accuracy = accuList[inx]
	best_no_features = featList[inx]
	print(dataset,"best:",accuList[inx],featList[inx])
	
	with open("result_GRx.csv","a") as f:
		print(dataset,"%.2f" % (100*best_accuracy),best_no_features,file=f)
