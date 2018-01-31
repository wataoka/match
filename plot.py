import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./result.tsv', delimiter='\t')
ax = data.plot()
fig = ax.get_figure()
fig.savefig('result.png', dpi=600)
