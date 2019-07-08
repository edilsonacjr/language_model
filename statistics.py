import pandas as pd

df = pd.read_csv('metadata.csv')

df.shape

df = df[df.apply(lambda x: not pd.isnull(x['language']) and 'en' in x['language'], axis=1)]

df.shape

df.drop_duplicates(['author','title']).groupby('author').count().sort_values('id', ascending=False)[['id']]

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#sns.set_style("whitegrid")

plt.rcParams["figure.figsize"] = [30, 20]

df_a = pd.read_csv('authors_tokens.csv')


#df_a['tokens'] = np.log(df_a['tokens'])

#plot = df_a.boxplot(column='tokens', by='author')
plot = sns.boxplot('author', 'tokens', orient='v', data=df_a )
plot = sns.swarmplot('author', 'tokens', data=df_a, color=".25" )

#plt.show()

fig = plot.get_figure()
fig.savefig("authors_tokens.png")



