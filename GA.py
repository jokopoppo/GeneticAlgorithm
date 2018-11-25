from pprint import pprint
import numpy as np
import random
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def check_accurancy(y):
    k=[]
    for i in y :
        if i[0]>i[1]:
            k.append(1)
        else:
            k.append(0)
    return np.asarray(k)

def fitness_function(train,validate):
    fitness = []
    for c in chromosome:
        wHidden = []
        for i in range(hidden):
            wHidden.append([])
            for j in range(train[0].__len__()):
                wHidden[-1].append(c[j + (i * train[0].__len__())])

        wOutput = []

        for i in range(output):
            wOutput.append([])
            for j in range(hidden):
                wOutput[-1].append(c[(train[0].__len__() * hidden) + (j + (i * hidden))])

        check = []
        for n in train:

            v = []
            y = []

            for i in range(hidden):
                v.append((np.asarray(wHidden[i]) * np.asarray(n)))
                v[-1] = sum(v[-1])
                y.append(sigmoid(v[-1]))

            v = []

            for i in range(output):
                v.append((np.asarray(wOutput[i]) * np.asarray(y)))
                v[-1] = sum(v[-1])

            y = softmax(v)
            check.append(y)
        check = check_accurancy(check)
        check = np.logical_xor(check, validate)
        fitness.append((check.__len__() - sum(check)))
    return fitness

file = open('wdbc.data','r')

data = []
for i in file :
    data.append(i.split(','))

validate = []
for i in data:
    i.pop(0)
    if(i[0] == 'M'):
        validate.append(1)
    else:
        validate.append(0)
    i.pop(0)


train = data
for i in range(train.__len__()):
    for j in range(train[i].__len__()) :
        train[i][j].strip('\n')
        train[i][j] = float(train[i][j])

output = 2
hidden = 3

chromosome = []
for i in range(50):
    chromosome.append([])
    for j in range((train[0].__len__()+ output) * hidden ):
        chromosome[-1].append(random.uniform(0, 1))


percent = round(train.__len__()*0.1)
best = None
maximun = 0
for fold in range(10):
    forTest = train[fold*percent:percent*fold+percent]
    validateTest = np.asarray(validate[fold*percent:percent*fold+percent])
    forTrain = train[:fold*percent] + train[fold*percent+percent:]
    validateTrain = np.asarray(validate[:fold*percent] + validate[fold*percent+percent:])
    for gen in range(1):
        fitness = fitness_function(forTrain,validateTrain)
        print("Generation",gen,"MAX",max(fitness)/forTrain.__len__(),"MEAN",np.mean(fitness)/forTrain.__len__())

        sector = fitness/sum(fitness)
        roulette = [random.uniform(0,1)]
        count = [0]
        for i in range(1,fitness.__len__()):
            sector[i] +=sector[i-1]
            roulette.append(random.uniform(0,1))
            count.append(0)

        for i in roulette:
            for j in range(sector.__len__()):
                if i < sector[j] :
                    count[j]+=1
                    break

        cross_pool = []
        for i in range(chromosome.__len__()):
            for j in range(count[i]):
                cross_pool.append(chromosome[i])

        mutate_pool = []
        for i in range(int(cross_pool.__len__()/2)):
            tmp = []
            tmp.append(cross_pool.pop(random.randrange(0,cross_pool.__len__(), 1)))
            tmp.append(cross_pool.pop(random.randrange(0,cross_pool.__len__(), 1)))
            cross_point = random.randrange(0, tmp[0].__len__(), 1)

            child1 = tmp[0][:cross_point] + tmp[1][cross_point:]
            child2 = tmp[1][:cross_point] + tmp[0][cross_point:]
            mutate_pool.append(child1)
            mutate_pool.append(child2)

        for i in range(mutate_pool.__len__()):
            if random.randrange(0, 100, 1)==1:
                c=i
                mutate_point = random.randrange(0, mutate_pool[i].__len__(), 1)
                n=mutate_point
                mutate_pool[i][mutate_point] += (random.uniform(-1,1))

        chromosome=mutate_pool

    acc = fitness_function(forTest,validateTest)
    print("Fold",fold,max(acc)/forTest.__len__())
    if(max(acc)/forTest.__len__() > maximun):
        maximun = max(acc)/forTest.__len__()
        best = chromosome[acc.index(max(acc))]


print("Accurancy =",maximun,"Weights :",best)