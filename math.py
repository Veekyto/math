import random
import copy
import numpy as np
import matplotlib.pyplot as plt

shortest_distance=[[0,0.7,1.4,1.9,3,3.5,0.6,1.2,1.8,2.5,3.1,3.9,0.9,2.4,2.9,3.2,4,2,3.7,4.4,5.3,5.4,2.5,4.5,5.1,5.3,5.9,6,3.3,4,4.6,5.2,5.6,5.9],
                   [0.7,0,0.7,1.2,2.3,2.8,1.1,0.5,1.1,1.8,2.4,3.2,1.4,1.7,2.2,2.5,3.3,2.5,3.5,4.2,4.6,4.7,3,4.3,4.9,5.1,5.4,5.3,3.8,4.5,5.1,5.7,6.1,5.8],
                   [1.4,0.7,0,0.5,1.6,2.1,1.8,1.2,0.4,1.1,1.7,2.5,2.1,2.4,1.5,1.8,2.6,3.2,4.2,3.7,3.9,4,3.7,5,4.4,4.6,4.7,4.6,4.5,5.2,5.4,5.2,5.4,5.1],
                   [1.9,1.2,0.5,0,1.1,1.6,2.3,1.7,0.9,0.9,1.4,2,2.6,2.9,1.3,1.6,2.4,3.7,4.7,4.2,3.7,3.8,4.2,5.5,4.9,5,4.5,4.4,5,5.7,5.9,5.6,5.2,4.9],
                   [3,2.3,1.6,1.1,0,0.5,3.3,2.7,1.6,0.9,0.3,0.9,3.6,3.6,1.3,1.4,1.8,4.7,5.2,4.5,3.5,3.2,5.2,6,5.2,4.8,4.3,3.8,6,6.2,5.6,5,4.6,4.3],
                   [3.5,2.8,2.1,1.6,0.5,0,3.8,3.2,2.1,1.4,0.8,0.4,4.1,4.1,1.8,1.9,1.3,5.2,5.3,4.6,3.4,2.7,5.7,5.9,4.9,4.4,3.8,3.3,6.4,5.7,5.1,4.5,4.1,3.8],
                   [0.6,1.1,1.8,2.3,3.3,3.8,0,0.6,1.7,2.4,3,3.9,0.3,1.8,2.8,3.1,3.9,1.4,3.1,3.8,5,5.3,1.9,3.9,4.5,4.7,5.3,5.8,2.7,3.4,4,4.6,5,5.3],
                   [1.2,0.5,1.2,1.7,2.7,3.2,0.6,0,1.1,1.8,2.4,3.3,0.9,1.2,2.2,2.5,3.3,2,3,3.7,4.6,4.7,2.5,3.8,4.4,4.6,5.2,5.3,3.3,4,4.6,5.2,5.6,5.8],
                   [1.8,1.1,0.4,0.9,1.6,2.1,1.7,1.1,0,0.7,1.3,2.2,2,2.3,1.1,1.4,2.2,3.1,4,3.3,3.5,3.6,3.6,4.8,4,4.2,4.3,4.2,4.4,5.1,5,4.8,5,4.7],
                   [2.5,1.8,1.1,0.9,0.9,1.4,2.4,1.8,0.7,0,0.6,1.5,2.7,2.7,0.4,0.7,1.5,3.8,4.3,3.6,2.8,2.9,4.3,5.1,4.3,4.1,3.6,3.5,5.1,5.8,5.3,4.7,4.3,4],
                   [3.1,2.4,1.7,1.4,0.3,0.8,3,2.4,1.3,0.6,0,0.9,3.3,3.3,1,1.1,1.8,4.4,4.9,4.2,3.2,3.2,4.9,5.7,4.9,4.5,4,3.8,5.7,6.2,5.6,5,4.6,4.3],
                   [3.9,3.2,2.5,2,0.9,0.4,3.9,3.3,2.2,1.5,0.9,0,4.2,4.2,1.9,1.7,0.9,5.3,4.9,4.2,3,2.3,5.8,5.5,4.5,4,3.4,2.9,6,5.3,4.7,4.1,3.7,3.4],
                   [0.9,1.4,2.1,2.6,3.6,4.1,0.3,0.9,2,2.7,3.3,4.2,0,1.5,3.1,3.4,4.2,1.1,2.8,3.5,4.7,5.4,1.6,3.6,4.2,4.4,5,5.5,2.4,3.1,3.7,4.3,4.7,5],
                   [2.4,1.7,2.4,2.9,3.6,4.1,1.8,1.2,2.3,2.7,3.3,4.2,1.5,0,2.3,2.6,3.4,2.6,1.8,2.5,3.7,4.4,3.1,2.6,3.2,3.4,4,4.5,3.9,3.6,4.2,4,4.4,4.7],
                   [2.9,2.2,1.5,1.3,1.3,1.8,2.8,2.2,1.1,0.4,1,1.9,3.1,2.3,0,0.3,1.1,4.2,3.9,3.2,2.4,2.5,4.7,4.7,3.9,3.7,3.2,3.1,5.5,5.5,4.9,4.3,3.9,3.6],
                   [3.2,2.5,1.8,1.6,1.4,1.9,3.1,2.5,1.4,0.7,1.1,1.7,3.4,2.6,0.3,0,0.8,4.5,4,3.3,2.1,2.2,5,4.8,3.9,3.4,2.9,2.8,5.8,5.2,4.6,4,3.6,3.3],
                   [4,3.3,2.6,2.4,1.8,1.3,3.9,3.3,2.2,1.5,1.8,0.9,4.2,3.4,1.1,0.8,0,5.3,4,3.3,2.1,1.4,5.8,4.6,3.6,3.1,2.5,2,5.1,4.4,3.8,3.2,2.8,2.5],
                   [2,2.5,3.2,3.7,4.7,5.2,1.4,2,3.1,3.8,4.4,5.3,1.1,2.6,4.2,4.5,5.3,0,1.7,2.4,3.6,4.3,0.5,2.5,3.1,3.3,3.9,4.4,1.3,2,2.6,3.2,3.6,3.9],
                   [3.7,3.5,4.2,4.7,5.2,5.3,3.1,3,4,4.3,4.9,4.9,2.8,1.8,3.9,4,4,1.7,0,0.7,1.9,2.6,2.2,0.8,1.4,1.6,2.2,2.7,2.5,1.8,2.4,2.2,2.6,2.9],
                   [4.4,4.2,3.7,4.2,4.5,4.6,3.8,3.7,3.3,3.6,4.2,4.2,3.5,2.5,3.2,3.3,3.3,2.4,0.7,0,1.2,1.9,2.9,1.5,0.7,0.9,1.5,2,3,2.3,1.7,1.5,1.9,2.2],
                   [5.3,4.6,3.9,3.7,3.5,3.4,5,4.6,3.5,2.8,3.2,3,4.7,3.7,2.4,2.1,2.1,3.6,1.9,1.2,0,0.7,4.1,2.7,1.8,1.3,0.8,1.3,3.8,3.1,2.5,1.9,1.9,1.8],
                   [5.4,4.7,4,3.8,3.2,2.7,5.3,4.7,3.6,2.9,3.2,2.3,5.4,4.4,2.5,2.2,1.4,4.3,2.6,1.9,0.7,0,4.5,3.2,2.2,1.7,1.1,0.6,3.7,3,2.4,1.8,1.4,1.1],
                   [2.5,3,3.7,4.2,5.2,5.7,1.9,2.5,3.6,4.3,4.9,5.8,1.6,3.1,4.7,5,5.8,0.5,2.2,2.9,4.1,4.5,0,2,3,3.3,3.9,3.9,0.8,1.5,2.1,2.7,3.1,3.4],
                   [4.5,4.3,5,5.5,6,5.9,3.9,3.8,4.8,5.1,5.7,5.5,3.6,2.6,4.7,4.8,4.6,2.5,0.8,1.5,2.7,3.2,2,0,1,1.5,2.1,2.6,1.7,1,1.6,2.1,2.5,2.8],
                   [5.1,4.9,4.4,4.9,5.2,4.9,4.5,4.4,4,4.3,4.9,4.5,4.2,3.2,3.9,3.9,3.6,3.1,1.4,0.7,1.8,2.2,3,1,0,0.5,1.1,1.6,2.3,1.6,1,1.1,1.5,1.8],
                   [5.3,5.1,4.6,5,4.8,4.4,4.7,4.6,4.2,4.1,4.5,4,4.4,3.4,3.7,3.4,3.1,3.3,1.6,0.9,1.3,1.7,3.3,1.5,0.5,0,0.6,1.1,2.5,1.8,1.2,0.6,1,1.3],
                   [5.9,5.4,4.7,4.5,4.3,3.8,5.3,5.2,4.3,3.6,4,3.4,5,4,3.2,2.9,2.5,3.9,2.2,1.5,0.8,1.1,3.9,2.1,1.1,0.6,0,0.5,3.1,2.4,1.8,1.2,1.1,1],
                   [6,5.3,4.6,4.4,3.8,3.3,5.8,5.3,4.2,3.5,3.8,2.9,5.5,4.5,3.1,2.8,2,4.4,2.7,2,1.3,0.6,3.9,2.6,1.6,1.1,0.5,0,3.1,2.4,1.8,1.2,0.8,0.5],
                   [3.3,3.8,4.5,5,6,6.4,2.7,3.3,4.4,5.1,5.7,6,2.4,3.9,5.5,5.8,5.1,1.3,2.5,3,3.8,3.7,0.8,1.7,2.3,2.5,3.1,3.1,0,0.7,1.3,1.9,2.3,2.6],
                   [4,4.5,5.2,5.7,6.2,5.7,3.4,4,5.1,5.8,6.2,5.3,3.1,3.6,5.5,5.2,4.4,2,1.8,2.3,3.1,3,1.5,1,1.6,1.8,2.4,2.4,0.7,0,0.6,1.2,1.6,1.9],
                   [4.6,5.1,5.4,5.9,5.6,5.1,4,4.6,5,5.3,5.6,4.7,3.7,4.2,4.9,4.6,3.8,2.6,2.4,1.7,2.5,2.4,2.1,1.6,1,1.2,1.8,1.8,1.3,0.6,0,0.6,1,1.3],
                   [5.2,5.7,5.2,5.6,5,4.5,4.6,5.2,4.8,4.7,5,4.1,4.3,4,4.3,4,3.2,3.2,2.2,1.5,1.9,1.8,2.7,2.1,1.1,0.6,1.2,1.2,1.9,1.2,0.6,0,0.4,0.7],
                   [5.6,6.1,5.4,5.2,4.6,4.1,5,5.6,5,4.3,4.6,3.7,4.7,4.4,3.9,3.6,2.8,3.6,2.6,1.9,1.9,1.4,3.1,2.5,1.5,1,1.1,0.8,2.3,1.6,1,0.4,0,0.3],
                   [5.9,5.8,5.1,4.9,4.3,3.8,5.3,5.8,4.7,4,4.3,3.4,5,4.7,3.6,3.3,2.5,3.9,2.9,2.2,1.8,1.1,3.4,2.8,1.8,1.3,1,0.5,2.6,1.9,1.3,0.7,0.3,0]
                   ]

capacity=[0,6,4,7,5,3,3,5,3,4,3,4,4,2,2,8,2,2,4,0,3,1,5,3,6,5,6,5,6,6,2,6,2,0]

class Robot:
    def __init__(self,location,speed):
        self.location=location
        self.speed=speed
        self.time=0



class Individual:
    def __init__(self):
        self.cost=0
        self.path=[]
        
#得到不包含充电桩(0,33)和图书馆(19)的初始随机路径
def init_population(pop_size=100,num=34):
    population=[]
    for i in range(pop_size):
        individual=Individual()
        for j in range(num):
            if j not in [0,19,33]:
                individual.path.append(j)
        random.shuffle(individual.path)
        population.append(individual)
    return population

#得到实际路径，机器人书量到达10则返回图书馆
def get_actual_path(lst, threshold=10):
    result = []
    temp_sum = 0
    for num in lst:
        temp_sum += capacity[num]
        if temp_sum > threshold:
            result.append(19)
            temp_sum = capacity[num]  
        result.append(num)
    return result

#得到每个个体的代价
def set_cost(individual,robot1,robot2):
    actual_path=get_actual_path(individual.path)
    robot1.time=0
    robot2.time=0
    leisure_robot=(lambda x, y: x if x.time < y.time else y)(robot1, robot2)
    for i in actual_path:
        leisure_robot.time+=shortest_distance[leisure_robot.location][i]/leisure_robot.speed
        leisure_robot.location=i
        if i==19:
            leisure_robot=(lambda x, y: x if x.time < y.time else y)(robot1, robot2)
    robot1.time+=shortest_distance[robot1.location][33]/robot1.speed
    robot2.time+=shortest_distance[robot2.location][0]/robot2.speed
    individual.cost=max(robot1.time,robot2.time)

def elitism_selection(population):
    sorted_population = sorted(population, key=lambda x: x.cost)
    quarter = len(sorted_population) // 4
    top_quarter = sorted_population[:quarter]
    middle_half = sorted_population[quarter: 3*quarter]
    bottom_quarter = sorted_population[3*quarter:]
    next_generation = copy.deepcopy(middle_half)
    next_generation.extend(top_quarter * 2)
    return next_generation



def roulette_wheel_selection(population):
    total_cost = sum(individual.cost for individual in population)
    selection_prob = [(total_cost - individual.cost) / total_cost for individual in population]
    prob_sum = sum(selection_prob)
    selection_prob = [prob / prob_sum for prob in selection_prob]
    num_individuals = len(population)
    next_generation = []
    for i in range(num_individuals):
        selected_index = random.choices(range(num_individuals), weights=selection_prob, k=1)[0]
        next_generation.append(copy.copy(population[selected_index]))
    return next_generation

def mutate(individual,robot1,robot2,it,mutation_rate=0.1):
    new_individual=copy.deepcopy(individual)
    if random.random() < mutation_rate:
        size=len(individual.path)
        i, j = random.sample(range(size), 2)
        new_individual.path[i], new_individual.path[j] = new_individual.path[j], new_individual.path[i]
        set_cost(individual,robot1,robot2)
        set_cost(new_individual,robot1,robot2)
        if new_individual.cost < individual.cost:
            return new_individual
        else:
            if random.random() < np.exp(-(new_individual.cost - individual.cost)*it):
                return new_individual
            else:
                return individual
    return new_individual   

def pmx_cross(parent1, parent2,rate=0.8):
    if random.random() > rate:
        return parent1, parent2
    size = len(parent1.path)
    point1, point2 = sorted(random.sample(range(size), 2))
    temp1 = parent1.path[point1:point2+1]
    temp2 = parent2.path[point1:point2+1]
    child1Path, child2Path = list(parent1.path), list(parent2.path)
    child1Path[point1:point2+1], child2Path[point1:point2+1] = temp2, temp1
    # 3. 解决冲突 - 将匹配区域外相同的基因置换
    for i in list(range(point1)) + list(range(point2+1, size)) :
        while child1Path[i] in temp2:
            index = temp2.index(child1Path[i])
            child1Path[i] = temp1[index]
        while child2Path[i] in temp1:
            index = temp1.index(child2Path[i])
            child2Path[i] = temp2[index]
    child1=Individual()
    child2=Individual()
    child1.path=child1Path
    child2.path=child2Path
    return child1, child2

def random_removal(individual):
    size=len(individual.path)
    random_int  = random.randint(0,size-1)
    remove=individual.path.pop(random_int)
    return remove
           

def worst_removal(individual):
    size=len(individual.path)
    cost = [0]*size
    cost[0] = shortest_distance[0][individual.path[0]]
    cost[size-1] = shortest_distance[individual.path[size-1]][19]
    for i in range(1,size -1):
        cost[i] = shortest_distance[individual.path[i]][individual.path[i-1]] + shortest_distance[individual.path[i]][individual.path[i+1]]
    remove = individual.path.pop(cost.index(max(cost)))
    return remove


def shaw_removal(individual,alpha=0.8):
    size=len(individual.path)
    similarity = [0]*size
    random_node = random.choice(individual.path)
    for i in range(size):
        if i != random_node:
            similarity[i] = alpha*shortest_distance[individual.path[i]][random_node]
            actual_path=get_actual_path(individual.path)
            index1=actual_path.index(random_node)
            index2=actual_path.index(individual.path[i])
            if index1>index2:
                index1,index2=index2,index1
            if 19 not in actual_path[index1:index2]:
                similarity[i]+=(1-alpha)
    remove = individual.path.pop(similarity.index(max(similarity)))
    return remove

def rebuild(individual,remove):
    size=len(individual.path)
    best_position = []
    best_cost = float('inf')
    new_individual=copy.copy(individual)
    for position in range(size+1):
        new_individual.path.insert(position, remove)
        set_cost(new_individual,robotA,robotB)
        cost=new_individual.cost
        new_individual.path.remove(remove)
        if cost < best_cost:
            best_position = position
            best_cost = cost
    new_individual.path.insert(best_position, remove)
    
    return new_individual
   

def removeAndRebuild(individual,alpha=0.5):
    a=random.random()
    if a<0.2:
        remove=random_removal(individual)
    elif a>=0.2 and a<=0.5:
        remove=shaw_removal(individual,alpha)
    else:
        remove=worst_removal(individual)
    new_individual=rebuild(individual,remove)
    return new_individual


            



robotA=Robot(location=33,speed=8)
robotB=Robot(location=0,speed=10)
best_cost=[]
iteration=[]
population=init_population()
for it in range(1,500):
    for individual in population:
        set_cost(individual,robotA,robotB)   
    sorted_population = sorted(population, key=lambda x: x.cost)
    if it ==499:
        print(get_actual_path(sorted_population[0].path))
    if it%10==0:
        best_cost.append(sorted_population[0].cost)
        iteration.append(it)
    population=elitism_selection(population)
    new_population=[]
    for _ in range(len(population) // 2):
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        child1,child2= pmx_cross(parent1, parent2)
        child1=mutate(child1,robotA,robotB,it)
        child2=mutate(child2,robotA,robotB,it)
        child1=removeAndRebuild(child1,alpha=0.5)
        child2=removeAndRebuild(child2,alpha=0.5)
        new_population.append(child1)
        new_population.append(child2)
    population=new_population


plt.plot(iteration,best_cost)
plt.show()
        

    







            






