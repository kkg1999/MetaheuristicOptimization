import numpy as np
import random
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math,time,sys
from matplotlib import pyplot
import pandas as pd
from datetime import datetime

epoch = 30 # parameter
pop_size = 20 # parameter
pp = 0.1 # parameter
A, epxilon = 4, 0.001
ID_MIN_PROBLEM = 0
ID_MAX_PROBLEM = -1
ID_POS = 0
ID_FIT = 1
omega = 0.9

def sigmoid1(gamma):
    #print(gamma)
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))


def initialise(partCount, dim, trainX, testX, trainy, testy):    
    population=np.zeros((partCount,dim))
    minn = 1
    maxx = math.floor(0.5*dim)
    fit = np.array([])
    if maxx<minn:
        maxx = minn + 1
        #not(c[i].all())
    
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
    #print(population.shape)
    for i in range(population.shape[0]):
        fit = np.append(fit, fitness(population[i], trainX, testX, trainy, testy))
    
    list_of_tuples = list(zip(population, fit))
        
    return list_of_tuples
    
def _get_global_best__( pop, id_fitness, id_best):
    minn = 100
    temp = pop[0]
    for i in pop:
        #print(i[1])
        minn = min(minn, i[1])
        temp = i
        
    return temp
    
def fitness(agent, trainX, testX, trainy, testy):
    # print(agent)
    cols=np.flatnonzero(agent)
    # print(cols)
    val=1
    if np.shape(cols)[0]==0:
        return val    
    clf=KNeighborsClassifier(n_neighbors=5)
    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=1-clf.score(test_data,testy)

    #in case of multi objective  []
    set_cnt=sum(agent)
    set_cnt=set_cnt/np.shape(agent)[0]
    val=omega*val+(1-omega)*set_cnt
    return val


def test_accuracy(agent, trainX, testX, trainy, testy):
    cols=np.flatnonzero(agent)
    val=1
    if np.shape(cols)[0]==0:
        return val    
    clf=KNeighborsClassifier(n_neighbors=5)
    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=clf.score(test_data,testy)
    return val

def onecnt(agent):
    return sum(agent)

        
def randomwalk(agent):
    percent = 30
    percent /= 100
    neighbor = agent.copy()
    size = np.shape(agent)[0]
    upper = int(percent*size)
    if upper <= 1:
        upper = size
    x = random.randint(1,upper)
    pos = random.sample(range(0,size - 1),x)
    for i in pos:
        neighbor[i] = 1 - neighbor[i]
    return neighbor


def adaptiveBeta(agent, trainX, testX, trainy, testy):
    bmin = 0.1 #parameter: (can be made 0.01)
    bmax = 1
    maxIter = 10 # parameter: (can be increased )
    
    agentFit = agent[1]
    agent = agent[0].copy()
    for curr in range(maxIter):
        neighbor = agent.copy()
        size = np.shape(neighbor)[0]
        neighbor = randomwalk(neighbor)

        beta = bmin + (curr / maxIter)*(bmax - bmin)
        for i in range(size):
            random.seed( time.time() + i )
            if random.random() <= beta:
                neighbor[i] = agent[i]
        neighFit = fitness(neighbor,trainX,testX,trainy,testy)
        if neighFit <= agentFit:
            agent = neighbor.copy()
            agentFit = neighFit
            


    return (agent,agentFit)

def sailFish(dataset):
    
        #url = "https://raw.githubusercontent.com/Rangerix/UCI_DATA/master/CSVformat/BreastCancer.csv"
        df = pd.read_csv(dataset)
        a, b = np.shape(df)
        data = df.values[:,0:b-1]
        label = df.values[:,b-1]
        dimension = data.shape[1]
        
        cross = 5
        test_size = (1/cross)
        trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
        clf=KNeighborsClassifier(n_neighbors=5)
        clf.fit(trainX,trainy)
        val=clf.score(testX,testy)
        whole_accuracy = val
        print("Total Acc: ",val)

        s_size = int(pop_size / pp)
        sf_pop = initialise(pop_size, dimension, trainX, testX, trainy, testy) 
        s_pop = initialise(s_size, dimension, trainX, testX, trainy, testy) 
        
        
        sf_gbest = _get_global_best__(sf_pop, ID_FIT, ID_MIN_PROBLEM)
        s_gbest = _get_global_best__(s_pop, ID_FIT, ID_MIN_PROBLEM)
        
        temp = np.array([])

        for iterno in range(0, epoch):
            print(iterno)
            ## Calculate lamda_i using Eq.(7)
            ## Update the position of sailfish using Eq.(6)
            for i in range(0, pop_size):
                PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
                lamda_i = 2 * np.random.uniform() * PD - PD
                sf_pop_arr = s_gbest[ID_POS] - lamda_i * ( np.random.uniform() *
                                        ( sf_gbest[ID_POS] + s_gbest[ID_POS] ) / 2 - sf_pop[i][ID_POS] )
                sf_pop_fit = sf_pop[i][ID_FIT]
                new_tuple = (sf_pop_arr, sf_pop_fit)
                
                sf_pop[i] = new_tuple
            ## Calculate AttackPower using Eq.(10)
            AP = A * ( 1 - 2 * (iterno) * epxilon )
            if AP < 0.5:
                alpha = int(len(s_pop) * AP )
                beta = int(dimension * AP)
                ### Random choice number of sardines which will be updated their position
                list1 = np.random.choice(range(0, len(s_pop)), alpha)
                for i in range(0, len(s_pop)):
                    if i in list1:
                        #### Random choice number of dimensions in sardines updated
                        list2 = np.random.choice(range(0, dimension), beta)
                        s_pop_arr = s_pop[i][ID_POS]
                        for j in range(0, dimension):
                            if j in list2:
                                ##### Update the position of selected sardines and selected their dimensions
                                s_pop_arr[j] = np.random.uniform()*( sf_gbest[ID_POS][j] - s_pop[i][ID_POS][j] + AP )
                        s_pop_fit = s_pop[i][ID_FIT]
                        new_tuple = ( s_pop_arr, s_pop_fit)
                        s_pop[i] = new_tuple
            else:
                ### Update the position of all sardine using Eq.(9)
                for i in range(0, len(s_pop)):
                    s_pop_arr = np.random.uniform()*( sf_gbest[ID_POS] - s_pop[i][ID_POS] + AP )
                    s_pop_fit = s_pop[i][ID_FIT]
                    new_tuple = (s_pop_arr, s_pop_fit)
                    s_pop[i] = new_tuple
            
            # population in binary
            # y, z = np.array([]), np.array([])
            # ychosen = 0
            # zchosen = 0
            # # print(np.shape(s_pop))
            for i in range(np.shape(s_pop)[0]):
                agent = s_pop[i][ID_POS]
                tempFit = s_pop[i][ID_FIT]
                random.seed(time.time())
                #print("agent shape :",np.shape(agent))
                y, z = np.array([]), np.array([])
                for j in range(np.shape(agent)[0]): 
                    random.seed(time.time()*200+999)
                    r1 = random.random()
                    random.seed(time.time()*200+999)
                    if sigmoid1(agent[j]) < r1:
                        y = np.append(y,0)
                    else:
                        y = np.append(y,1)

                yfit = fitness(y, trainX, testX, trainy, testy)
                agent = deepcopy(y)
                tempFit = yfit
                
                new_tuple = (agent,tempFit)
                s_pop[i] = new_tuple
            ## Recalculate the fitness of all sardine
            # print("y chosen:",ychosen,"z chosen:",zchosen,"total: ",ychosen+zchosen)
            for i in range(0, len(s_pop)):
                s_pop_arr = s_pop[i][ID_POS]
                s_pop_fit = fitness(s_pop[i][ID_POS],trainX, testX, trainy, testy)
                new_tuple = (s_pop_arr, s_pop_fit)
                s_pop[i] = new_tuple

            # local search algo
            for i in range(np.shape(s_pop)[0]):
                new_tuple = adaptiveBeta(s_pop[i],trainX,testX,trainy,testy)
                s_pop[i] = new_tuple

            ## Sort the population of sailfish and sardine (for reducing computational cost)
            sf_pop = sorted(sf_pop, key=lambda temp: temp[ID_FIT])
            s_pop = sorted(s_pop, key=lambda temp: temp[ID_FIT])
            for i in range(0, pop_size):
                s_size_2 = len(s_pop)
                if s_size_2 == 0:
                    s_pop = initialise(s_pop, dimension, trainX, testX, trainy, testy)
                    s_pop = sorted(s_pop, key=lambda temp: temp[ID_FIT])
                for j in range(0, s_size):
                    ### If there is a better solution in sardine population.
                    if sf_pop[i][ID_FIT] > s_pop[j][ID_FIT]:
                        sf_pop[i] = deepcopy(s_pop[j])
                        del s_pop[j]
                    break   #### This simple keyword helped reducing ton of comparing operation.
                            #### Especially when sardine pop size >> sailfish pop size
            
            # OBL
            # sf_pop = OBL(sf_pop, trainX, testX, trainy, testy)
            sf_current_best = _get_global_best__(sf_pop, ID_FIT, ID_MIN_PROBLEM)
            s_current_best = _get_global_best__(s_pop, ID_FIT, ID_MIN_PROBLEM)
            if sf_current_best[ID_FIT] < sf_gbest[ID_FIT]:
                sf_gbest = np.array(deepcopy(sf_current_best))
            if s_current_best[ID_FIT] < s_gbest[ID_FIT]:
                s_gbest = np.array(deepcopy(s_current_best))
            
        
        testAcc = test_accuracy(sf_gbest[ID_POS], trainX, testX, trainy, testy)
        featCnt = onecnt(sf_gbest[ID_POS])
        print("Test Accuracy: ", testAcc)
        print("#Features: ", featCnt)

        return sf_gbest[ID_POS], testAcc, featCnt

    
datasetlist = ["BreastCancer.csv", "BreastEW.csv", "CongressEW.csv", "Exactly.csv", "Exactly2.csv", "HeartEW.csv", "Ionosphere.csv",  "Lymphography.csv"]
datasetlist = ["M-of-n.csv", "PenglungEW.csv", "Sonar.csv", "SpectEW.csv", "Tic-tac-toe.csv", "Vote.csv", "Wine.csv", "Zoo.csv"]
# datasetname =  sys.argv[1]  
# "KrVsKpEW.csv", "WaveformEW.csv",
for datasetname  in datasetlist:    
    print(datasetname)
    accuArr = []
    featArr = []
    start_time = datetime.now()
    for i in range(20):
        # print(i)
        agentBest, testAcc, featCnt = sailFish("csvUCI/"+datasetname)
        # print(testAcc)
        accuArr.append(testAcc)
        featArr.append(featCnt)
    time_required = datetime.now() - start_time
    maxx = max(accuArr)
    currFeat= 20000
    for i in range(np.shape(accuArr)[0]):
        if accuArr[i]==maxx and featArr[i] < currFeat:
            currFeat = featArr[i]
    datasetname = datasetname.split('.')[0]
    print(datasetname)
    print(maxx,currFeat)
    print("time_required:",time_required)
    with open("result_BSF1.csv","a") as f:
        print(datasetname,maxx,currFeat,time_required,file=f)
# print(sf_gbest)
#print(temp)
#print(loss_train)