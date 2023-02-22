
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from collections import Counter
import sys
import os

plt.rcParams["figure.figsize"] = (40, 20)
from itertools import chain
import tqdm as tqdm


data = pd.read_excel('Hack 13.6 tỷ tại tech (fb).xlsx', sheet_name=2)

G = nx.from_pandas_edgelist(data,
                            source='Source Name',
                            target='Target Name',
                            edge_attr='point',
                            create_using=nx.DiGraph())

weighted_degrees = dict(nx.degree(G, weight='point'))
print(weighted_degrees)

ax = plt.gca()
fig = plt.gcf()
plt.axis('off')
plt.title('Network FB', fontsize=24)

pos = nx.spring_layout(G)

trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform

tick_params = {'top': 'off', 'bottom': 'off', 'left': 'off', 'right': 'off',
               'labelleft': 'off', 'labelbottom': 'off'}  #flag grid params
# styles = ['dotted','dashdot','dashed','solid'] # line styles


# draw edges
for e in G.edges(data=True):
    print(e)
    width = e[2]['point']  #normalize by max points
    nx.draw_networkx_edges(G, pos, edgelist=[e], width=width, style='solid')
    # in networkx versions >2.1 arrowheads can be adjusted

    #draw nodes
    # for node in G.nodes():
    #     imsize = max((0.3*G.in_degree(node,weight='points')/max(dict(G.in_degree(weight='points')).values()))**2,0.03)
    #     # size is proportional to the votes
    #
    #     (x,y) = pos[node]
    #     xx,yy = trans((x,y)) # figure coordinates
    #     xa,ya = trans2((xx,yy)) # axes coordinates
    #
    #     country = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ], label='test')
    #     # them 1 truc vao hinh hien tai va bien no thanh truc hien tai.
    #     # chen anh vao node
    #     country.set_aspect('equal')
    #     # ti le cua node. equal, auto, 0.7
    #     country.tick_params(**tick_params)

for node in G.nodes():
    size = 1000 * weighted_degrees[node]
    ns = nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size, node_color='#009fe3')
    ns.set_edgecolor('#f2f6fa')
nx.draw_networkx_labels(G, pos, font_size=15, font_color='red');
plt.show()

fig.savefig('images/network-fb.png')