import random
import math
import matplotlib.pyplot as plt
import tkinter as tk

class GeneticAlgorithm:
    def __init__(self, num_cities, generations, pop_size, crossover_rate, mutation_rate):
        self.num_cities = num_cities
        self.pop_size = pop_size
        self.generations = generations
        self.city_locations = self.generate_city_locations()
        self.population = self.generate_initial_population()
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def generate_city_locations(self):
        """Generate random locations for cities within a 10x10 grid"""
        return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(self.num_cities)]

    def generate_initial_population(self):
        """Generate an initial population of candidate solutions"""
        population = []
        for _ in range(self.pop_size):
            chromosome = list(range(1, self.num_cities+1))  # start from city 1 to 10
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def calculate_distance(self, city1, city2):
        """Calculate the Euclidean distance between two cities"""
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

    def calculate_fitness(self, chromosome):
        """Calculate the fitness of a candidate solution"""
        total_distance = 0
        for i in range(len(chromosome)):
            city1 = self.city_locations[chromosome[i]-1]  # adjust for 0-indexing
            city2 = self.city_locations[chromosome[(i+1) % len(chromosome)]-1]  # adjust for 0-indexing
            total_distance += self.calculate_distance(city1, city2)
        fitness = 1 / (total_distance + 1)  # calculate fitness as the inverse of the total distance
        return fitness

    def roulette_wheel_selection(self):
        """Perform roulette wheel selection to choose two parents"""
        fitnesses = [self.calculate_fitness(chromosome) for chromosome in self.population]
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        return random.choices(self.population, weights=probabilities, k=2)

    def ordered_crossover(self, parent1, parent2):
        """Perform ordered crossover (OX) to generate an offspring"""
        pos1 = random.randint(0, len(parent1) - 1)
        pos2 = random.randint(pos1, len(parent1) - 1)
        offspring1 = [-1] * len(parent1)
        offspring2 = [-1] * len(parent1)
        offspring1[pos1:pos2+1] = parent1[pos1:pos2+1]
        offspring2[pos1:pos2+1] = parent2[pos1:pos2+1]
        remaining_cities1 = [city for city in parent2 if city not in offspring1]
        remaining_cities2 = [city for city in parent1 if city not in offspring2]
        for i in range(len(offspring1)):
            if offspring1[i] == -1:
                offspring1[i] = remaining_cities1.pop(0)
            if offspring2[i] == -1:
                offspring2[i] = remaining_cities2.pop(0)
        return offspring1, offspring2

    def swap_mutation(self, chromosome):
        """Perform swap mutation on a candidate solution"""
        pos1 = random.randint(0, len(chromosome) - 1)
        pos2 = random.randint(0, len(chromosome) - 1)
        chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
        return chromosome

    def evolve_population(self):
        """Evolve the population over a number of generations"""
        for _ in range(self.generations):
            new_population = []
            for j in range(len(self.population)):
                parent1, parent2 = self.roulette_wheel_selection()
                offspring1, offspring2 = self.ordered_crossover(parent1, parent2) if random.random() < self.crossover_rate else (parent1, parent2)
                offspring1 = self.swap_mutation(offspring1) if random.random() < self.mutation_rate else offspring1
                offspring2 = self.swap_mutation(offspring2) if random.random() < self.mutation_rate else offspring2
                new_population.extend([offspring1, offspring2])
            fitnesses = [(self.calculate_fitness(chromosome), chromosome) for chromosome in new_population]
            fitnesses.sort(reverse=True)
            self.population = [chromosome for _, chromosome in fitnesses[:self.pop_size]]
        # Return the chromosome with the highest fitness (i.e., the shortest distance)
        return max([(self.calculate_fitness(chromosome), chromosome) for chromosome in self.population])

    def plot_cities(self):
        """Plot the locations of the cities on a scatter plot"""
        x, y = zip(*self.city_locations)
        plt.scatter(x, y)
        for i, loc in enumerate(self.city_locations):
            plt.annotate(str(i+1), loc)  # adjust for 1-indexing
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

    def plot_shortest_route(self, shortest_route):
        """Plot the shortest route found by the genetic algorithm"""
        self.plot_cities()
        for i in range(len(shortest_route)):
            city1 = self.city_locations[shortest_route[i]-1]  # adjust for 0-indexing
            city2 = self.city_locations[shortest_route[(i+1) % len(shortest_route)]-1]  # adjust for 0-indexing
            plt.plot([city1[0], city2[0]], [city1[1], city2[1]])
        plt.title('Solving the TSP using a genetic algorithm')
        plt.show()

    def run_algorithm(self):
        """Runs the algorithm and prints the shortest route and total distance, and plots the shortest route."""
        solution = self.evolve_population()
        shortest_distance = 1 / solution[0]  # calculate the shortest distance as the inverse of the fitness
        shortest_route = solution[1]
        print(f'Shortest route: {shortest_route}')
        print(f'Total distance: {shortest_distance}')
        self.plot_shortest_route(shortest_route)


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.num_cities_label = tk.Label(self, text="Number of cities:")
        self.num_cities_label.pack()
        self.num_cities_entry = tk.Entry(self)
        self.num_cities_entry.pack()

        self.generations_label = tk.Label(self, text="Number of generations:")
        self.generations_label.pack()
        self.generations_entry = tk.Entry(self)
        self.generations_entry.pack()

        self.pop_size_label = tk.Label(self, text="Population size:")
        self.pop_size_label.pack()
        self.pop_size_entry = tk.Entry(self)
        self.pop_size_entry.pack()

        self.crossover_rate_label = tk.Label(self, text="crossover_rate:")
        self.crossover_rate_label.pack()
        self.crossover_rate_entry = tk.Entry(self)
        self.crossover_rate_entry.pack()

        self.mutation_rate_label = tk.Label(self, text="mutation_rate:")
        self.mutation_rate_label.pack()
        self.mutation_rate_entry = tk.Entry(self)
        self.mutation_rate_entry.pack()

        self.start_button = tk.Button(self, text="Run the Genetic Algorithm",  fg="blue", command=self.run_algorithm)
        self.start_button.pack()

        self.quit_button = tk.Button(self, text="Quit", fg="red", command=self.master.destroy)
        self.quit_button.pack()

    def run_algorithm(self):
        num_cities = int(self.num_cities_entry.get())
        pop_size = int(self.pop_size_entry.get())
        generations = int(self.generations_entry.get())
        crossover_rate = float(self.crossover_rate_entry.get())
        mutation_rate = float(self.mutation_rate_entry.get())
        ga = GeneticAlgorithm(num_cities, generations, pop_size, crossover_rate, mutation_rate)
        ga.run_algorithm()

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("250x280")
    app = Application(root)
    root.mainloop()