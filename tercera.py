from random import randint
from tkinter import *
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

country = None
master = None
w = None
ga = None
fig = None
ax = None
ax2 = None
iteration = 0
a = []
a2= []

# mutation probability equals to 0.35 to avoid crossed path
class Chromosome:
    def __init__(self, size, mutation_probability = 0.35, data = [0]):
        if data == [0]:
            self.data = data
            for i in range(size-1):
                index = randint(1, size-1)
                while index in self.data:
                    index = randint(1, size-1)
                self.data.append(index)
            self.data.append(0)
        else:
            self.data = data
        
        self.mutation_probability = mutation_probability
        self.life_time = 0
    
    def crossover(self, other): ##Funcion crossover 1
        crossover_point = randint(1, len(self.data)-2)
        child1 = self.data[:crossover_point]
        for i in other.data:
            if i not in child1:
                child1.append(i)
        child1.append(0)
        child2 = other.data[:crossover_point]
        for i in self.data:
            if i not in child2:
                child2.append(i)
        child2.append(0)
        return child1, child2
    
    #ini fungsi mutasi
    def mutation(self):
        if randint(0, 100) <= self.mutation_probability*100:
            pos = randint(1, len(self.data)-2)
            size = randint(1, len(self.data)-1-pos)
            where = randint(1, len(self.data)-1-size)
            arre = [];
            for i in range(size) :
            	arre.append(self.data[pos])
            	self.data.pop(pos)
            self.data[where:where]= arre;
    
    def adaptability(self):
        distance = 0
        for i in range(len(self.data)-1):
            distance += sqrt((country.cities[self.data[i+1]].x-country.cities[self.data[i]].x)**2 + (country.cities[self.data[i+1]].y-country.cities[self.data[i]].y)**2)
        return distance


class GA:
    def __init__(self, country, max_generations = 1000, init_population_size = 100, max_population_size = 500, max_life_time = 3):
        self.country = country
        self.max_generations = max_generations
        self.max_life_time = max_life_time
        self.max_population_size = max_population_size
        self.population = []
        for i in range(init_population_size):
            self.population.append(Chromosome(country.size()))
        self.best = self.population[0].data
        self.best_equal_count = 0

    def step(self):
        self.sort()
        if self.best == self.get_best_2().data:
            self.best_equal_count += 1
        else:
            self.best_equal_count = 0
            self.best = self.get_best_2().data
        self.crossover()
        self.mutation()
        for i in self.population:                                                                                                                                                                                                                                                           
            if i.life_time > self.max_life_time:
                self.population.remove(i)
            else:
                i.life_time += 1

    def sort(self):
        self.population = sorted(self.population, key=Chromosome.adaptability)
        self.population = self.population[:self.max_population_size]

    def crossover(self):
        for i in range(0, len(self.population)-4, 2):
            child1_data, child2_data = self.population[i].crossover(self.population[i+1])
            child1 = Chromosome(country.size(), data = child1_data)
            child1.mutation()
            child2 = Chromosome(country.size(), data = child2_data)
            child2.mutation()
            self.population.append(child1)
            self.population.append(child2)

    def adaptability(self):
        adap = []
        for i in self.population:
            adap.append(i.adaptability())
        return adap

    def mutation(self):
        for i in self.population:
            i.mutation()

    def get_best(self): ##Funcion selectora 1
        return [p for (a, p) in sorted(zip(self.adaptability(), self.population))][0]
    
    def get_best_2(self) : ##Funcion selectora 2
        return sorted(self.population, key=Chromosome.adaptability)[0]


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return str(self.x) + ", " + str(self.y)


class Country:
    def __init__(self, w, h, rand = 0, cities = []):
        self.w = w
        self.h = h
        ciudades = pd.read_csv("ciudades.csv", header=None)
        c1 = ciudades[0]
        c2 = ciudades[1]
        if rand == 0:
            self.cities = cities
        else:
            self.cities = []
            for i in range(rand):
                self.add_random_city(int(c1[i+1]),int(c2[i+1]))
    
    def add_city(self, c):
        self.cities.append(c)
    
    def add_random_city(self,w,h):
        self.cities.append(City(w, h))
    
    def size(self):
        return len(self.cities)
    
    def __str__(self):
        s = ""
        for i in self.cities:
            s = s + "[" + str(i) + "]" + ", "
        return s[:-2]


def step():
    global iteration
    print(iteration)
    iteration += 1
    
    w.delete(ALL)
    for city in country.cities:
        w.create_oval(city.x-2, city.y-2, city.x+2, city.y+2, fill="red")
    
    best = ga.get_best_2().data
    fitness = 1/ float(ga.get_best_2().adaptability())
    a.append(ga.get_best_2().adaptability())
    a2.append(fitness)
    for i in range(len(best)-1):
        w.create_line(country.cities[best[i]].x, country.cities[best[i]].y, country.cities[best[i+1]].x, country.cities[best[i+1]].y, fill="blue")
    
    ga.step()
    GraficarDistancia(a)
    ax2.set_title("Histograma de Distancias i = "+str(iteration))
    ax.set_title("Diagrama de Ganancia de Distancia max = "+str(ga.get_best_2().adaptability()))
    
    if ga.best_equal_count > 99:
        print("Solution found")
        GraficarHistograma(a2)    
    elif iteration > ga.max_generations:
        print("Max generations reached")
    else:
        master.after(1, step)


def GraficarDistancia(arreglo):
    ax.clear()
    ax2.clear()
    ax.bar(np.arange(iteration),arreglo)
    fig.canvas.draw()


def GraficarHistograma(arreglo):
    ax2.hist(arreglo)
    fig.canvas.draw()


def main():
    global country, master, w, ga ,a ,fig, ax, ax2
    fig = plt.figure(1)
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plt.ion()
    fig.show()
    fig.canvas.draw()
    
    country = Country(700, 700, 20)
    master = Tk()
    w = Canvas(master, width=700, height=700)
    w.pack()
    
    ga = GA(country)
    
    master.after(1, step)
    master.mainloop()


if __name__ == '__main__':
    main()
