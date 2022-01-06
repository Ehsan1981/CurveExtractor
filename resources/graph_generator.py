import matplotlib.pyplot as plt
import numpy as np
import numpy.random


# Parameters
FILENAME = "./placeholder.png"
TITLE = "CurveFinder"
XLABEL = "Time t [s]"
YLABEL = r"Temperature T [$\degree$]"
STYLE = "fivethirtyeight"
NUM_CURVE = 2
SEED = 1851895
RANDOM_SEED = False

# Data generation
if RANDOM_SEED:
    SEED = np.random.randint(0, 3000000)
    print(SEED)

np.random.seed(SEED)

x = np.linspace(0, 10, 100)
y = [np.zeros(x.shape) for i in range(NUM_CURVE)]

for data in y:
    for i in range(5):
        amp = np.random.uniform(-1, 1)
        freq = np.random.uniform(0.1, 2)
        is_sin = np.random.choice([True, False])
        data += amp*np.sin(freq*x) if is_sin else amp*np.cos(freq*x)

# Generating the plot
plt.rcParams.update({'font.size': 20})
plt.style.use(STYLE)
plt.figure()
for data in y:
    plt.plot(x, data)
plt.grid(True)
plt.title(TITLE)
plt.xlabel(XLABEL)
plt.ylabel(YLABEL)
plt.savefig(FILENAME, dpi=150, bbox_inches="tight")
