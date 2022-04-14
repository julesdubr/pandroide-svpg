# Cette cellule contient le code complet de l'expérience de navigation dans le maze.
# Complétez les trous dans les balises <ANSWER></ANSWER>


import gym, gym_fastsim

from deap import *
import numpy as np
from nn import SimpleNeuralControllerNumpy
from scipy.spatial import KDTree

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import argparse

import array
import random
import operator
import math

from plot import *

from scoop import futures

from novelty_search import *

weights = (-1.0, 1.0)

if hasattr(creator, "MyFitness"):
    # Deleting any previous definition (to avoid warning message)
    print("Main: destroying myfitness")
    del creator.MyFitness
creator.create("MyFitness", base.Fitness, weights=(weights))

if hasattr(creator, "Individual"):
    # Deleting any previous definition (to avoid warning message)
    del creator.Individual
creator.create("Individual", list, fitness=creator.MyFitness)

# Evaluation d'un réseau de neurones en utilisant gym
def eval_nn(genotype, nbstep=2000, render=False, name=""):
    """Evaluation d'une politique parametrée par le génotype

    Evaluation d'une politique parametrée par le génotype
    :param genotype: le paramètre de politique à évaluer
    :param nbstep: le nombre maximum de pas de temps
    :param render: affichage/sauvegarde de la trajectoire du robot
    :param name: nom à donner au fichier de log
    """

    # Cette fonction fait l'évaluation d'un genotype utilisé pour
    # paraméter un réseau de neurones de structure fixe.
    # En vue de l'expérience avec novelty_search, elle renvoie:
    # - la fitness: distance à la sortie à minimiser (qui peut se récupérer dans la 4eme valeur de retour de la méthode step, qui est un dictionnaire dans lequel la clé "dist_obj" donne accès à l'opposé de la distance à la sortie)
    # - le descripteur comportemental, qui est la position finale qui est accessible depuis le même dictionnaire (clé "robot_pos", dont il ne faut garder que les 2 premières composantes)
    nn = SimpleNeuralControllerNumpy(5, 2, 2, 10)
    nn.set_parameters(genotype)
    observation = env.reset()
    old_pos = None
    total_dist = 0
    if render:
        f = open("traj" + name + ".log", "w")

    # <ANSWER>

    # </ANSWER>


def nsga2_NS(
    n=100,
    nbgen=200,
    IND_SIZE=10,
    variant="FIT",
    MIN_V=-30,
    MAX_V=30,
    CXPB=0.6,
    MUTPB=0.3,
    verbose=False,
):

    # votre code contiendra donc des tests comme suit pour gérer la différence entre ces variantes:
    if hasattr(creator, "MyFitness"):
        # Deleting any previous definition (to avoid warning message)
        print("nsga2_NS: destroying myfitness")
        del creator.MyFitness

    # Attention: le signe du poids associé à la fitness doit être adapté aux valeurs renvoyées par eval_nn (négatif si valeur à minimiser et positif si à maximiser)
    if variant == "FIT+NS":
        print("Creator: FIT+NS")
        creator.create("MyFitness", base.Fitness, weights=(-1.0, 1.0))
    elif variant == "FIT":
        print("Creator: FIT")
        creator.create("MyFitness", base.Fitness, weights=(-1.0,))
    elif variant == "NS":
        print("Creator: NS")
        creator.create("MyFitness", base.Fitness, weights=(1.0,))
    else:
        print("Variante inconnue: " + variant)

    if hasattr(creator, "Individual"):
        # Deleting any previous definition (to avoid warning message)
        del creator.Individual
    creator.create("Individual", list, fitness=creator.MyFitness)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.uniform, MIN_V, MAX_V)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attribute,
        n=IND_SIZE,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "mate", tools.cxSimulatedBinaryBounded, eta=15, low=MIN_V, up=MAX_V
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=15,
        low=MIN_V,
        up=MAX_V,
        indpb=1 / IND_SIZE,
    )
    toolbox.register("select", tools.selNSGA2, k=n)
    toolbox.register("select_dcd", tools.selTournamentDCD, k=n)
    toolbox.register("evaluate", eval_nn, render=True)

    toolbox.register("map", futures.map)

    population = toolbox.population(n=n)
    paretofront = tools.ParetoFront()

    # Permet de sauvegarder les descripteurs comportementaux de tous les individus rencontrés.
    # vous pouvez tracer les descripteurs générés depuis une cellule de jupyter avec la commande:
    # plot_points_file("bd.log")
    fbd = open("bd.log", "w")

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses_bds = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, (fit, bd) in zip(invalid_ind, fitnesses_bds):
        # print("Fit: "+str(fit))
        # print("BD: "+str(bd))

        # Complétez pour positionner fitness.values selon la variante choisie
        # <ANSWER>

        # </ANSWER>

        fbd.write(" ".join(map(str, bd)) + "\n")
        fbd.flush()

    if paretofront is not None:
        paretofront.update(population)

    # print("Pareto Front: "+str(paretofront))

    k = 15
    add_strategy = "random"
    lambdaNov = 6

    # Crée l'archive et mets à jour les champs ind.novelty de chaque individu
    archive = updateNovelty(population, population, None, k, add_strategy, lambdaNov)

    # Selon la variante, complétez ind.fitness.values pour ajouter la valeur de ind.novelty qui vient d'être calculée
    # <ANSWER>

    # </ANSWER>

    indexmin, valuemin = max(
        enumerate([i.fit for i in population]), key=operator.itemgetter(1)
    )

    # Pour le calcul des crowdingDistances si utilisation de selTournamentDCD
    population[:] = toolbox.select(population)

    # Begin the generational process
    for gen in range(1, nbgen + 1):
        if gen % 10 == 0:
            print("+", end="", flush=True)
        else:
            print(".", end="", flush=True)

        # Complétez avec un NSGA-2 en prenant soin de mettre à jour les calculs de nouveauté et d'ajouter les
        # nouveautés à la fitness des individus.

        # ATTENTION: dans la mise à jour de la nouveauté (updateNovelty), vérifiez bien la signification du premier et du 2eme argument.
        # l'appel fait plus haut doit être adapté: la nouveauté de TOUS les individus doit être recalculée (parents+enfants)
        # et les individus susceptibles d'être ajoutés à l'archive ne doivent être pris que parmi les enfants.

        # <ANSWER>

        # </ANSWER>

        # Update the hall of fame with the generated individuals
        if paretofront is not None:
            paretofront.update(population)

        # used to track the max value (useful in particular if using only novelty)
        indexmin, newvaluemin = min(
            enumerate([i.fit for i in pq]), key=operator.itemgetter(1)
        )
        if newvaluemin < valuemin:
            valuemin = newvaluemin
            print(
                "Gen "
                + str(gen)
                + ", new min ! min fit="
                + str(valuemin)
                + " index="
                + str(indexmin)
            )
            eval_nn(pq[indexmin], True, "gen%04d" % (gen))
    fbd.close()
    return population, None, paretofront


env = gym.make("FastsimSimpleNavigation-v0")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch maze navigation experiment.")
    parser.add_argument("--nbgen", type=int, default=200, help="number of generations")
    parser.add_argument("--popsize", type=int, default=100, help="population size")
    parser.add_argument(
        "--res_dir",
        type=str,
        default="res",
        help="basename of the directory in which to put the results",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="FIT",
        choices=["FIT", "NS", "FIT+NS"],
        help="variant to consider",
    )

    # Il vous est recommandé de gérer les différentes variantes avec cette variable. Les 3 valeurs possibles seront:
    # "FIT+NS": expérience multiobjectif avec la fitness et la nouveauté (NSGA-2)
    # "NS": nouveauté seule
    # "FIT": fitness seule
    # pour les variantes avec un seul objectif, il vous est cependant recommandé d'utiliser NSGA-2 car cela limitera la différence entre les variantes et cela
    # vous fera gagner du temps pour la suite.

    args = parser.parse_args()
    print("Number of generations: " + str(args.nbgen))
    nbgen = args.nbgen
    print("Population size: " + str(args.popsize))
    popsize = args.popsize
    print("Variant: " + args.variant)
    variant = args.variant

    nn = SimpleNeuralControllerNumpy(5, 2, 2, 10)
    IND_SIZE = len(nn.get_parameters())

    pop, logbook, paretofront = nsga2_NS(
        n=popsize, variant=variant, nbgen=nbgen, IND_SIZE=IND_SIZE
    )
    # plot_pareto_front(paretofront, "Final pareto front")
    for i, p in enumerate(paretofront):
        print("Visualizing indiv " + str(i) + ", fit=" + str(p.fitness.values))
        eval_nn(p, True)

    env.close()
