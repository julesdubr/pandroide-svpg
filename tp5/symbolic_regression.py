from deap import creator, gp, base, tools, algorithms
import operator
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import pickle
import datetime
import sys
import os


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def ru():
    return random.uniform(-1, 1)


def evalSymbReg(individual, input, output, nb_obj=1):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    sqerrors = []
    for i in range(len(input)):
        sqerrors.append((func(*input[i]) - output[i]) ** 2)
    if nb_obj == 1:
        return (math.fsum(sqerrors) / len(sqerrors),)
    else:
        return math.fsum(sqerrors) / len(sqerrors), len(individual)


if __name__ == "__main__":

    random.seed()

    parser = argparse.ArgumentParser(description="Launch symbolic regression run.")

    parser.add_argument("--nb_gen", type=int, default=200, help="number of generations")
    parser.add_argument("--mu", type=int, default=400, help="population size")
    parser.add_argument(
        "--lambda_", type=int, default=400, help="number of individuals to generate"
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        default="res",
        help="basename of the directory in which to put the results",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="elitist",
        choices=["elitist", "double_tournament", "nsga2"],
        help="selection scheme",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="f1",
        choices=["f1", "f2"],
        help="function to fit",
    )

    # for question 1.2
    parser.add_argument(
        "--noise",
        type=float,
        default="0.",
        help="noise added to the model to fit (gaussian, mean=0, sigma=noise)",
    )

    args = parser.parse_args()
    print("Number of generations: " + str(args.nb_gen))
    ngen = args.nb_gen
    print("Population size: " + str(args.mu))
    mu = args.mu
    print("Number of offspring to generate: " + str(args.lambda_))
    lambda_ = args.lambda_
    print("Selection scheme: " + str(args.selection))
    sel = args.selection
    if sel == "nsga2":
        nb_obj = 2
    else:
        nb_obj = 1
    print("Basename of the results dir: " + str(args.res_dir))
    name = args.res_dir

    if nb_obj == 1:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    elif nb_obj == 2:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))

    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    noise = args.noise
    problem = args.problem

    d = datetime.datetime.today()
    if name != "":
        sep = "_"
    else:
        sep = ""
    run_name = name + "_" + sel + "_" + d.strftime(name + sep + "%Y_%m_%d-%H-%M-%S")
    try:
        os.makedirs(run_name)
    except OSError:
        pass

    print("Putting the results in : " + run_name)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(lambda ind: ind.height)
    mstats = tools.MultiStatistics(
        fitness=stats_fit, size=stats_size, height=stats_height
    )
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    if problem == "f1":
        nb_dim = 2
        input_training = []
        output_training = []
        input_testing = []
        output_testing = []
        name_vars = {"ARG0": "x1", "ARG1": "x2"}

        # Complétez pour générer l'ensemble d'entrainement et de validation avec une fonction choisie à 2 dimensions
        # <ANSWER>

        # </ANSWER>
    # en OPTION: vous pouvez faire des tests sur d'autres fonctions
    elif problem == "f2":
        # <ANSWER>
        pass

        # </ANSWER>

    pset = gp.PrimitiveSet("MAIN", nb_dim)

    # Complétez pour constituer l'ensemble de primitives qui pourront être utilisées
    # <ANSWER>

    # </ANSWER>

    pset.addTerminal(1)
    pset.addEphemeralConstant("cst", ru)
    pset.renameArguments(**name_vars)

    cxpb = 0.5
    mutpb = 0.1

    toolbox = base.Toolbox()

    # En vous inspirant des exemples de programmation génétique dans DEAP,
    # enregistrez les différents opérateurs que vous utiliserez dans la suite.
    # Vous choisirez l'opérateur de sélection en fonction de la variable sel
    # (voir valeurs possibles dans le parser d'arguments)
    # <ANSWER>

    # </ANSWER>

    pop = toolbox.population(n=400)

    if nb_obj == 1:
        print("Hall-of-fame: best solution")
        hof = tools.HallOfFame(1)
    else:
        print("Hall-of-fame: Pareto front")
        hof = tools.ParetoFront()

    # Pour simplifier, plutôt que d'écrire la boucle, vous pourrez utiliser un algorithme tout intégré,
    # par exemple eaMuPlusLambda (cf https://deap.readthedocs.io/en/master/api/algo.html).
    # Cela ne permettra pas de générer un NSGA-II complet, mais cela vous permettra de faire de premiers tests.
    # En option, si vous avez le temps, vous pourrez tester un NSGA-II complet pour voir si cela change les résultats.
    # <ANSWER>

    # </ANSWER>

    log = tools.Logbook()
    # Affichage des résultats. Tout est dans le répertoire run_name
    avg, dmin, dmax = log.chapters["fitness"].select("avg", "min", "max")
    gen = log.select("gen")

    plt.figure()
    plt.yscale("log")
    plt.plot(gen[1:], dmin[1:])
    plt.title("Minimum error")
    plt.savefig(run_name + "/min_error_gen%d.pdf" % (ngen))

    plt.figure()
    plt.yscale("log")
    plt.fill_between(gen[1:], dmin[1:], dmax[1:], alpha=0.25, linewidth=0)
    plt.plot(gen[1:], avg[1:])
    plt.title("Average error")
    plt.savefig(run_name + "/avg_error_gen%d.pdf" % (ngen))

    avg, dmin, dmax = log.chapters["size"].select("avg", "min", "max")
    gen = log.select("gen")
    plt.figure()
    plt.yscale("log")
    plt.fill_between(gen[1:], dmin[1:], dmax[1:], alpha=0.25, linewidth=0)
    plt.plot(gen[1:], avg[1:])
    plt.title("Average size")
    plt.savefig(run_name + "/avg_size_gen%d.pdf" % (ngen))

    avg, dmin, dmax = log.chapters["height"].select("avg", "min", "max")
    gen = log.select("gen")
    plt.figure()
    plt.yscale("log")
    plt.fill_between(gen[1:], dmin[1:], dmax[1:], alpha=0.25, linewidth=0)
    plt.plot(gen[1:], avg[1:])
    plt.title("Average height")
    plt.savefig(run_name + "/avg_height_gen%d.pdf" % (ngen))

    with open(run_name + "/pset_gen%d.npz" % (ngen), "wb") as f:
        pickle.dump(pset, f)

    for i, ind in enumerate(hof):
        print("=========")
        print("HOF %d, len=%d" % (i, len(ind)))
        print(
            "Error on the training dataset: %f"
            % (evalSymbReg(ind, input_training, output_training, nb_obj=1))
        )
        print(
            "Error on the testing dataset: %f"
            % (evalSymbReg(ind, input_testing, output_testing, nb_obj=1))
        )
        with open(run_name + "/hof%d_gen%d.npz" % (i, ngen), "wb") as f:
            pickle.dump(ind, f)

        nodes, edges, labels = gp.graph(ind)

        ### Graphviz Section ###
        import pygraphviz as pgv

        plt.figure()

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for ni in nodes:
            n = g.get_node(ni)
            n.attr["label"] = labels[ni]

        g.draw(run_name + "/hof%d_tree_gen%d.pdf" % (i, ngen))

    print("Results saved in " + run_name)
