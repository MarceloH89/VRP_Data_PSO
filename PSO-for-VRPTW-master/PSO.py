import os
import sys
import math
import random
import functools
from matplotlib import pyplot as plt
import csv
import copy

class PSO_VRPTW:
    trunk_amount = None  
    trunk_volumes = None  
    target_amount = None  
    target_sites = None  
    target_time_limits = None  
    target_volumes = None 
    target_service_times = None  
    dist = None  
    losses = []

    omega = 0.4  
    c1, c2 = 0.1, 0.5  
    n = 10  
    dot_v = None  
    dot_bests = None 
    dot_solutions = None  
    best_solution = None  

    color = ['#00FFFF', '#7FFFD4', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887', '#FFFF00', '#9ACD32', '#008000',
             '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#DC143C', '#00FFFF', '#008B8B', '#B8860B',
             '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A',
             '#8FBC8F', '#483D8B', '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF',
             '#B22222', '#FFFAF0', '#228B22', '#FF00FF', '#DCDCDC', '#F8F8FF', '#FFD700', '#DAA520', '#808080',
             '#FF69B4', '#CD5C5C', '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5', '#7CFC00', '#FFFACD',
             '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#90EE90', '#D3D3D3', '#FFB6C1', '#FFA07A', '#20B2AA',
             '#87CEFA', '#778899', '#B0C4DE', '#FFFFE0', '#00FF00', '#32CD32', '#FAF0E6', '#FF00FF', '#800000',
             '#66CDAA', '#0000CD', '#BA55D3', '#9370DB', '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585',
             '#191970', '#F5FFFA', '#FFE4E1', '#FFE4B5', '#FFDEAD', '#000080', '#FDF5E6', '#808000', '#6B8E23',
             '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE', '#DB7093', '#FFEFD5', '#FFDAB9',
             '#CD853F', '#FFC0CB', '#DDA0DD', '#B0E0E6', '#800080', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513',
             '#FA8072', '#FAA460', '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0', '#87CEEB', '#6A5ACD', '#708090',
             '#FFFAFA', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE',
             '#F5DEB3', '#FFFFFF', '#F5F5F5']
    rate = 0.10

    def __init__(self, trunk_volumes: list, target_sites: list
                 , target_time_limits: list, target_volumes: list, target_service_times):
        self.trunk_volumes = trunk_volumes
        self.target_sites = target_sites
        self.target_time_limits = target_time_limits
        self.target_volumes = target_volumes
        self.target_service_times = target_service_times
        self.trunk_amount = len(trunk_volumes)
        self.target_amount = len(target_volumes)
        self.n = 10 * len(target_service_time)  
        self.dot_solutions = [[[], []] for i in range(self.n)] 
        self.dot_bests = [None for i in range(self.n)]  
        self.dot_v = [None for i in range(self.n)] 
        self.dist = [(lambda x: [math.sqrt(math.pow(self.target_sites[0][x] - self.target_sites[0][y], 2)
                                           + math.pow(self.target_sites[1][x] - self.target_sites[1][y], 2))
                                 for y in range(self.target_amount)])(i)
                     for i in range(self.target_amount)]  

    def cost(self, solution):
        """ Calcular el costo de la solución. """
        PE, PL = 10, 100  
        max_time = 0  
        sum_dist = 0  
        penalty = 0 
        for k in range(self.trunk_amount):  
            pre_pos, pre_time = 0, 0 
            k_time = 0  
            counter = 0
            for i in range(1, len(solution[0])):
                if solution[0][i] == k:  
                    k_time += self.dist[pre_pos][i]
                    sum_dist += self.dist[pre_pos][i]
                    counter += 1
                    if self.target_time_limits[0][i] > k_time and counter > 1:  
                        k_time = self.target_time_limits[0][i]
                        penalty += PE * (self.target_time_limits[0][i] - k_time)
                    elif self.target_time_limits[1][i] < k_time:  
                        penalty += PL * (k_time - self.target_time_limits[1][i])
                    k_time += self.target_service_times[i]
                    pre_pos = i  
            k_time += self.dist[pre_pos][0]  
            sum_dist += self.dist[pre_pos][0]
            max_time = max_time if k_time < max_time else k_time
        return sum_dist + penalty  

    def recode_solution(self, solution: list):
        temp = copy.deepcopy(solution[0])
        i, mapp = 0, dict()
        counter = [1 for i in range(self.trunk_amount)]
        for i in range(self.target_amount):
            if solution[0][i] not in mapp:
                mapp[solution[0][i]] = len(mapp)
        for i in range(self.target_amount):  
            solution[1][i] = counter[mapp[solution[0][i]]]
            counter[mapp[solution[0][i]]] += 1
            solution[0][i] = mapp[solution[0][i]]
        return

    def init_solution(self):
        for i in range(self.n):
            self.dot_solutions[i][0].clear()
            self.dot_solutions[i][1].clear()
            self.dot_solutions[i][0].append(0)  
            self.dot_solutions[i][1].append(0)  
            mapp = dict()
            order = [0 for i in range(self.trunk_amount)]
            volumes = [0 for i in range(self.trunk_amount)]  
            for j in range(1, self.target_amount):
                No_ = random.randint(0, self.trunk_amount - 1) 
                counter = 0
                if No_ not in mapp: 
                    mapp[No_] = len(mapp)
                while volumes[No_] + self.target_volumes[j] > self.trunk_volumes[No_]:  
                    No_ = random.randint(0, self.trunk_amount - 1)
                    counter += 1
                    if No_ not in mapp:  
                        mapp[No_] = len(mapp)
                    if counter > 0xffff:
                        print('La capacidad del vehículo es gravemente insuficiente para cargar toda la mercancía, y no hay solución')
                        return -1
                self.dot_solutions[i][0].append(mapp[No_])  
                order[mapp[No_]] += 1  
                self.dot_solutions[i][1].append(order[mapp[No_]]) 
                volumes[No_] += self.target_volumes[j] 

    def update_best_solution(self):
        min_cost = 0x7fffffff if self.best_solution is None else self.cost(self.best_solution)
        pos, i = 0, 0
        while i < len(self.dot_solutions):
            if self.cost(self.dot_solutions[i]) < min_cost:
                min_cost = self.cost(self.dot_solutions[i])
                pos = i  
            i += 1
        self.losses.append(min_cost)
        if pos != 0:
            self.best_solution = copy.deepcopy(self.dot_solutions[pos])

    def update_dot_best(self):
        for i in range(self.n):
            if (self.dot_bests[i] is None) or self.cost(self.dot_solutions[i]) < self.cost(self.dot_bests[i]):
                self.dot_bests[i] = copy.deepcopy(self.dot_solutions[i])

    def draw_pic(self):
        plt.subplot(2, 2, 1)
        for i in range(self.trunk_amount):
            plt.scatter(self.target_sites[0][0], self.target_sites[1][0], color='', marker='o', edgecolors='b', s=50)
            pre = 0
            for j in range(1, len(self.best_solution[0])):
                if self.best_solution[0][j] == i:
                    plt.plot([self.target_sites[0][pre], self.target_sites[0][j]],
                             [self.target_sites[1][pre], self.target_sites[1][j]],
                             color=self.color[i])
                    pre = j
            plt.plot([self.target_sites[0][0], self.target_sites[0][pre]],
                     [self.target_sites[1][0], self.target_sites[1][pre]],
                     color=self.color[i])
        plt.title('Cost=' + str(self.cost(self.best_solution)))
        plt.subplot(2, 2, 2)  
        plt.plot(self.losses)
        plt.subplot(2, 1, 2)
        for k in range(self.trunk_amount):
            y = k * 10  
            time, pre, counter = 0, 0, 0
            for j in range(1, self.target_amount):
                if self.best_solution[0][j] == k:
                    time += self.dist[pre][j]
                    time = time if self.target_time_limits[0][j] < time else self.target_time_limits[0][j]
                    plt.scatter(time, y, color='', marker='o', edgecolors='b', s=50)
                    time += self.target_service_times[j]
                    plt.plot([self.target_time_limits[0][j], self.target_time_limits[1][j]], [y, y]
                             , color='r', linewidth=2.5)
                    plt.scatter(self.target_time_limits[0][j], y, color='', marker='o'
                                , edgecolors=self.color[counter], s=35)
                    plt.scatter(self.target_time_limits[1][j], y, color='', marker='o'
                                , edgecolors=self.color[counter], s=35)
                    pre = j
                    counter += 1
            plt.plot([0, time], [y, y], color='b', linewidth=1)
        plt.show()

    def dot_improve(self):
        """ En cada ronda, las partículas se aproximan a la solución óptima. """
        sum_factor = self.omega + self.c1 + self.c2
        selection = [self.omega / sum_factor, (self.omega + self.c1) / sum_factor, 1]
        for i in range(self.n):
            temp_volumes = [0 for i in range(self.trunk_amount)]  
            for j in range(self.target_amount):
                num = random.random()
                if num > selection[1]:
                    self.dot_solutions[i][0][j] = self.best_solution[0][j]
                elif num > selection[2]:
                    self.dot_solutions[i][0][j] = self.dot_bests[i][0][j]
                if random.random() < self.rate:  
                    self.dot_solutions[i][0][j] = random.randint(0, len(self.trunk_volumes) - 1)
                while temp_volumes[self.dot_solutions[i][0][j]] + self.target_volumes[j] \
                        > self.trunk_volumes[self.dot_solutions[i][0][j]]:
                    self.dot_solutions[i][0][j] = random.randint(0, len(self.trunk_volumes) - 1)
                temp_volumes[self.dot_solutions[i][0][j]] += self.target_volumes[j]

    def run(self):
        """ Entrada del algoritmo de enjambre de partículas """
        self.init_solution()
        self.dot_bests = copy.deepcopy(self.dot_solutions)  
        self.update_best_solution()  
        for epoch in range(400):
            if epoch % 10 == 0:
                print('Best cost:', self.cost(self.best_solution))
            if epoch % 100 == 0:
                self.rate /= 2
            self.dot_improve()
            for i in self.dot_solutions:
                self.recode_solution(i)
            self.update_dot_best()
            self.update_best_solution()
        print("Best solution:\n", self.best_solution[0], '\n', self.best_solution[1])
        self.draw_pic()


def read_in_data(path: str, row_num):
    def func_cmp(a, b):
        if a[3] == b[3]:
            return a[4] - b[4]
        return a[3] - b[3]
    data = [[[], []], [], [[], []], []]  
    data_in = []
    with open(path) as file:
        csvf = csv.reader(file)
        csvf.__next__() 
        for i in range(row_num):
            line = csvf.__next__()  
            ele = [float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])]
            data_in.append(ele)
    data_in = sorted(data_in, key=functools.cmp_to_key(func_cmp))
    for i in data_in:
        data[0][0].append(i[0])  
        data[0][1].append(i[1])  
        data[1].append(i[2])  
        data[2][0].append(i[3])  
        data[2][1].append(i[4])  
        data[3].append(i[5])  
    return data


if __name__ == "__main__":
    data = read_in_data('VRP_Data_Pso/PSO/input/Test.csv', 51)  
    trunk_volume = [200 for i in range(13)]  
    target_site = data[0]
    target_volume = data[1]
    target_time_limit = data[2]
    target_service_time = data[3]
    pso = PSO_VRPTW(trunk_volume, target_site, target_time_limit, target_volume, target_service_time)
    pso.run()