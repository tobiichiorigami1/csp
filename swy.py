import sys
import numpy as np
import matplotlib.pyplot as plt

def res(x):
    y=20+x[0]**2+x[1]**2 - 10*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))
    return y

w=1.0
c1=1.49445
c2=1.49445


maxgen = 200
sizepop = 20

Vmax = 1
Vmin =-1
popmax = 3
popmin =-5

pop = 5 * np.random.uniform(-1,1,(2,sizepop))
v=np.random.uniform(-1,1,(2,sizepop))

fitness = res(pop)
i =np.argmin(fitness)

gbest =pop
zbest =pop[:,i]
fitnessgbest = fitness
fitnesszbest = fitness[i]


t = 0
record = np.zeros(maxgen)
while t<maxgen:
    v=w*v+c1*np.random.random()*(gbest-pop) + c2 * np.random.random()*(zbest.reshape(2,1)-pop)
    v[v>Vmax]=Vmax
    v[v<Vmin]=Vmin

    pop = pop + 0.5 * v
    pop[pop >popmax]=popmax
    pop[pop < popmin]=popmin

    fitness = res(pop)

    index = fitness < fitnessgbest
    fitnessgbest[index] = fitness[index]
    gbest[:,index] = pop[:,index]
    j = np.argmin(fitness)
    if fitness[j] < fitnesszbest:
        zbest = pop[:,j]
        fitnesszbest = fitness[j]

    record[t] = fitnesszbest # 记录群体最优位置的变化   
    
    t = t + 1
    
print(zbest)
plt.plot(record,'b-')
plt.xlabel('generation')  
plt.ylabel('fitness')  
plt.title('fitness curve')  
plt.show()
    
