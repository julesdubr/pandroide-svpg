import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os


d = datetime.datetime.today()
directory = d.strftime(str(Path(__file__).parents[1]) + "/archives/%m-%d_%H-%M/")

if not os.path.exists(directory):
    os.makedirs(directory)

# plt.plot(range(5), np.random.randn(5))
# plt.plot(range(5), np.random.randn(5))
# plt.savefig(directory + "test.png")