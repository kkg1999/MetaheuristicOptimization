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
def sigmoid1(gamma):     #convert to probability
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


def fitness(position):
	cols=np.flatnonzero(position)
	val=1
	if np.shape(cols)[0]==0:
		return val	
	# clf = RandomForestClassifier(n_estimators=300)
	clf=KNeighborsClassifier(n_neighbors=5)
	# clf=MLPClassifier( alpha=0.01, max_iter=1000) #hidden_layer_sizes=(1000,500,100)
	#cross=3
	#test_size=(1/cross)
	#X_train, X_test, y_train, y_test = train_test_split(trainX, trainy,  stratify=trainy,test_size=test_size)
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


def allfit(population):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i])     
		#print(acc[i])
	return acc

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

def toBinary(population,popSize,dimension,oldPop):

	for i in range(popSize):
		for j in range(dimension):
			temp = Vfunction3(population[i][j])

			# if temp > 0.5: # sfunction
			# 	population[i][j] = 1
			# else:
			# 	population[i][j] = 0

			if temp > 0.5: # vfunction
				population[i][j] = (1 - oldPop[i][j])
			else:
				population[i][j] = oldPop[i][j]
	return population


#####################################################################################
omega = 0.85 #weightage for no of features and accuracy
popSize = 20
max_iter = 30
S = 2


# df=pd.read_csv(sys.argv[1])
# (a,b)=np.shape(df)
# print(a,b)
# data = df.values[:,0:b-1]
# label = df.values[:,b-1]
# dimension = np.shape(data)[1] #particle dimension


best_accuracy = -1
best_no_features = -1
average_accuracy = 0
global_count = 0
accuracy_list = []
features_list = []

for train_iteration in range(11):

	#---------------------------------------------------------------------
	#I know I should not put not it here, but still ...
	df=pd.read_csv(sys.argv[1])
	(a,b)=np.shape(df)
	print(a,b)
	data = df.values[:,0:b-1]
	label = df.values[:,b-1]
	dimension = np.shape(data)[1] #particle dimension
	#---------------------------------------------------------------------

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

	# for population_iteration in range(2):
	global_count += 1
	print('global: ',global_count)

	x_axis = []
	y_axis = []

	population = initialize(popSize,dimension)
	# print(population)

	start_time = datetime.now()
	fitList = allfit(population)
	bestInx = np.argmin(fitList)
	fitBest = min(fitList)
	Mbest = population[bestInx].copy()
	for currIter in range(max_iter):

		popnew = np.zeros((popSize,dimension))
		x_axis.append(currIter)
		y_axis.append(min(fitList))
		for i in range(popSize):
			random.seed(time.time() + 10.01)
			randNo = random.random()
			if randNo<0.5 :
				#chain foraging
				random.seed(time.time())
				r = random.random()
				alpha = 2*r*(abs(math.log(r))**0.5)
				if i == 1:
					popnew[i] = population[i] + r * (Mbest - population[i]) + alpha*(Mbest - population[i])
				else:
					popnew[i] = population[i] + r * (population[i-1] - population[i]) + alpha*(Mbest - population[i])
			else:
				#cyclone foraging
				cutOff = random.random()
				r = random.random()
				r1 = random.random()
				beta = 2 * math.exp(r1 * (max_iter - currIter + 1) / max_iter) * math.sin(2 * math.pi * r1)
				if currIter/max_iter < cutOff:
					# exploration
					Mrand = np.zeros(np.shape(population[0]))
					no = random.randint(1,max(int(0.1*dimension),2))
					random.seed(time.time()+ 100)
					pos = random.sample(range(0,dimension-1),no)
					for j in pos:
						Mrand[j] = 1

					if i==1 :
						popnew[i] = Mrand + r * (Mrand - population[i]) + beta * (Mrand - population[i])
					else:
						popnew[i] = Mrand + r * (population[i-1] - population[i]) + beta * (Mrand - population[i])
				else:
					# exploitation
					if i == 1:
						popnew[i] = Mbest + r * (Mbest - population[i]) + beta * (Mbest - population[i])
					else:
						popnew[i] = Mbest + r * (population[i-1] - population[i]) + beta * (Mbest - population[i])

		# print(popnew)
		
		popnew = toBinary(popnew,popSize,dimension,population)
		popnewTemp = popnew.copy()
		#compute fitness for each individual
		fitList = allfit(popnew)
		if min(fitList)<fitBest :
			bestInx = np.argmin(fitList)
			fitBest = min(fitList)
			Mbest = popnew[bestInx].copy()
		# print(fitList,fitBest)

		#somersault foraging
		for i in range(popSize):
			r2 = random.random()
			random.seed(time.time())
			r3 = random.random()
			popnew[i] = popnew[i] + S * (r2*Mbest - r3*popnew[i])

		popnew = toBinary(popnew,popSize,dimension,popnewTemp)
		#compute fitness for each individual
		fitList = allfit(popnew)
		if min(fitList)<fitBest :
			bestInx = np.argmin(fitList)
			fitBest = min(fitList)
			Mbest = popnew[bestInx].copy()
		# print(fitList,fitBest)

		population = popnew.copy()


	time_required = datetime.now() - start_time

	# pyplot.plot(x_axis,y_axis)
	# pyplot.xlim(0,max_iter)
	# pyplot.ylim(max(0,min(y_axis)-0.1),min(max(y_axis)+0.1,1))
	# pyplot.show()


	output = Mbest.copy()
	print(output)

	#test accuracy
	cols=np.flatnonzero(output)
	#print(cols)
	X_test=testX[:,cols]
	X_train=trainX[:,cols]
	#print(np.shape(feature))

	# clf = RandomForestClassifier(n_estimators=300)
	clf=KNeighborsClassifier(n_neighbors=5)
	#clf=MLPClassifier( alpha=0.001, max_iter=2000) #hidden_layer_sizes=(1000,500,100 ),
	clf.fit(X_train,trainy)
	val=clf.score(X_test, testy )
	print(val,onecount(output))

	accuracy_list.append(val)
	features_list.append(onecount(output))
	if ( val == best_accuracy ) and ( onecount(output) < best_no_features ):
		best_accuracy = val
		best_no_features = onecount( output )
		best_time_req = time_required
		best_whole_accuracy = whole_accuracy

	if val > best_accuracy :
		best_accuracy = val
		best_no_features = onecount( output )
		best_time_req = time_required
		best_whole_accuracy = whole_accuracy

print('best: ',best_accuracy, best_no_features)
# print('avg: ',average_accuracy/10)


# accuracy_list = np.array(accuracy_list)
# accuracy_list.sort()
# accuracy_list = accuracy_list[-4:]
# average = np.mean(accuracy_list)
# stddev = np.std(accuracy_list)

# accuracy_list = list(accuracy_list)
# avgFea = 0
# for i in accuracy_list:
# 	inx = accuracy_list.index(i)
# 	avgFea += features_list[inx]
# avgFea /= 4

temp=sys.argv[1].split('/')[-1]
with open("../Result/result_MRFOv3_uci20.csv","a") as f:
	print(temp,"%.2f" % (100*best_whole_accuracy) ,
		np.shape(df)[1] - 1,"%.2f" % (100*best_accuracy),best_no_features,file=f)