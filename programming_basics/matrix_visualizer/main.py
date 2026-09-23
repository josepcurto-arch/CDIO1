import matplotlib.pyplot as plt
import matplotlib
import numpy as np


matplotlib.use("QtAgg")


foo = np.random.randn(10, 10)
plt.imshow(foo)
plt.show()
