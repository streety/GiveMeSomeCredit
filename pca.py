import load_data

import numpy as np
import matplotlib.pyplot as plt
import mdp

data = load_data.load_data()
load_data.basic_stats(data)

dataset = load_data.prepare_dataset(data[1])



pcan = mdp.nodes.PCANode(output_dim=0.95)
pcar = pcan.execute(dataset[0])
print pcan.d

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pcar[:,0], pcar[:,1], 'k.')
plt.show()