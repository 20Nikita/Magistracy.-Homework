import pygmo
algo = pygmo.algorithm(maco(gen=100))
algo.set_verbosity(20)
pop = pygmo.population(zdt(1), 63)
pop = algo.evolve(pop) 