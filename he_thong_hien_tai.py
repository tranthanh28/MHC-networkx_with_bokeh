import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings
import random
import matplotlib.patches as patches

from collections import Counter
import sys
import os

plt.rcParams["figure.figsize"] = (20, 10)
from itertools import chain
import tqdm as tqdm

# from colorthief import ColorThief

warnings.filterwarnings('ignore')

# SỬA: sưa lại tên file và sheet
dataRoot = pd.read_excel('Hack 13.6 tỷ tại tech (fb).xlsx', sheet_name=1)

# for index, row in data.iterrows():
#     if row['Source Name'] == row['Target Name']:
#         print(row['Source Name'], row['Target Name'])
#         newData = data.drop(row[index], axis=1)
# print(newData)

# Xử lý bai post không bị lặp chính mình.
# dataPost = {}
# for index, row in dataRoot.iterrows():
#     if row['TYPE'] == 'POST':
#         dataPost[row['Source User']] = row['REACH']

# print(dataPost)
# hanlde data
# dataFilter = dataFilter.drop(dataFilter[(dataFilter.TTT < 5) & (dataFilter.TYPE == 'COMMENT')].index)
dataRoot = dataRoot.sort_values(by=['TTT'], ascending=False)
dataFilter = dataRoot.drop(dataRoot[(dataRoot.TYPE == 'COMMENT')].index)
# print(dataFilter)
data = dataFilter.groupby('MESSAGE').head()
# print(data.shape)
data = data.groupby((data['USER NAME'] + data['PAGE NAME']), as_index=False).agg(
    {'PAGE NAME': 'min', 'PAGE ID': 'min', 'USER ID': 'min', 'USER NAME': 'min', 'TYPE': 'min', 'TTT': 'sum'})
# SỬA: xóa bài post dưới x số lượng tương tác
# data = data.drop(data[(data.TTT < 10000) & (data.TYPE == 'POST')].index)

# SỬA: Chỉ lấy 5 bài viết nhiều tương tác nhất
data1 = data.sort_values(by=['TTT'], ascending=False)
dataPost = data1[data1.TYPE == 'POST'][0:5]
# print('dataPost')
# print(dataPost)
listPost = list(dataPost['PAGE NAME'])

dataComment = pd.DataFrame()
for index, row in dataPost.iterrows():
    # SỬA: Lấy ra 5 comment nhiều tương tác nhất tương ứng vs các bài Post
    dataComment = dataComment.append(
        dataRoot.loc[(dataRoot['PAGE NAME'] == row['USER NAME']) & (dataRoot['TYPE'] == 'COMMENT')][0:5])
    # if row['USER NAME'] == row['PAGE NAME']:
    #     dataPost.at[index, 'TTT'] = row['TTT'] / 2
# print(dataComment.shape)
data = dataPost.append(dataComment)

# make link_url for tap nodes
linkUser = {}
for index, row in data.iterrows():
    linkUser[row['PAGE NAME']] = 'https://facebook.com/' + str(row['PAGE ID'])
    linkUser[row['USER NAME']] = 'https://facebook.com/' + str(row['USER ID'])

G = nx.from_pandas_edgelist(data,
                            source='USER NAME',
                            target='PAGE NAME',
                            edge_attr='TTT',
                            create_using=nx.DiGraph())
nx.set_node_attributes(G, linkUser, "linkUser")

weighted_degrees = dict(G.in_degree(weight='TTT'))

ax = plt.gca()
fig = plt.gcf()
ax.margins(0.20)
plt.axis('off')
plt.title('Network FB', fontsize=24)

# pos = nx.spring_layout(G)
pos = nx.spring_layout(G, k=0.5, iterations=60, scale=10)

trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform

# tick_params = {'top': 'off', 'bottom': 'off', 'left': 'off', 'right': 'off',
#                'labelleft': 'off', 'labelbottom': 'off'}  #flag grid params
# styles = ['dotted','dashdot','dashed','solid'] # line styles


# draw edges
for e in G.edges(data=True):
    # width = max(e[2]['TTT'], 1) / 100  #normalize by max points
    nx.draw_networkx_edges(G, pos, edgelist=[e], width=1, style='solid')
    # in networkx versions >2.1 arrowheads can be adjusted

    # draw nodes
# for node in G.nodes():
#     print(node)
#     print(G.in_degree(node, weight='Point'))
#     imsize = max((0.3 * G.in_degree(node, weight='points')
#                   / max(dict(G.in_degree(weight='points')).values())) ** 2, 0.02)
#     print(imsize)
#     # size is proportional to the votes
#     # flag = mpl.image.imread(flags[node])
#
#     (x, y) = pos[node]
#     xx, yy = trans((x, y))  # figure coordinates
#     xa, ya = trans2((xx, yy))  # axes coordinates
#
#     # matplotlib.pyplot.axes
#     # them 1 truc vao hinh hien tai va bien no thanh truc hien tai.
#     # [left, bottom, width, height]
#     country = plt.axes([xa - imsize / 2.0, ya - imsize / 2.0, imsize, imsize])
#
#     # chen anh vao node
#     country.imshow(img[random.randint(0, N - 1)])
#     # ti le cua node. equal, auto, 0.7
#     country.set_aspect('equal')
#     # print(imsize/2)
#
#     # make cicle avatar
#     # patch = patches.Circle((xa, ya), radius=imsize/2.0, transform=ax.transData)
#     # country.set_clip_path(patch)
#     # country.tick_params(**tick_params)
#     country.axis('off')

# def nudge(poss):
#     return { n: (x, y - (100 * max(weighted_degrees[n], 1)**0.5)*0.0003) for n,(x,y) in poss.items()}
node_size = []
max_weight = max(weighted_degrees, key=lambda x: weighted_degrees[x])
for node in G.nodes():
    # print(node)
    # printweighted_degrees[node])
    size = 100 * max(weighted_degrees[node], 1) ** 0.5
    size = max(size, 10)
    a = weighted_degrees[node] * 100 / weighted_degrees[max_weight]
    node_size.append(max(a, 10))
    G.nodes[node]['node_size'] = max(a, 10)
    # size = max(max(weighted_degrees[node], 1) ** 0.5, 50)
    # print(size)
    ns = nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size, node_color='#009fe3')
    ns.set_edgecolor('#f2f6fa')
    # nx.draw_networkx_labels(G, pos, {node}, font_color='red');
# posNode = nudge(pos)

nx.draw_networkx_labels(G, pos, font_size=16, font_color='r')
fig.savefig('images/network-fb.png', facecolor='white')

from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool,
                          ColumnDataSource, OpenURL, TapTool,
                          PointDrawTool, MultiSelect, Column, LabelSet,
                          Select, PanTool, PolySelectTool, WheelZoomTool, SaveTool)
from bokeh.plotting import from_networkx, figure, output_file, show, curdoc
from bokeh.layouts import column, row
from bokeh.palettes import Spectral4

global graph

# node_size = {k:max(100 * max(v, 1) ** 0.5,400) for k,v in G.degree().items()}
# print(node_size)
# nx.set_node_attributes(G, node_size, name="node_size")
# source = ColumnDataSource(pd.DataFrame.from_dict({k: v for k, v in G.nodes(data=True)}, orient='index'))
# print(source)
# print({k: v for k, v in G.nodes(data=True)})

p = figure(width=1500, height=800, tools="tap", title="FB Network")

network_multiseclect = MultiSelect(title='An cac bai viet', options=[(i, i) for i in listPost], value=[''])


def dropNode(GNew, network, degree):
    # print('drop node')
    # print(network)
    # print(degree)
    if network != ['']:
        # print(network)
        for line in network:
            if not p.select_one({"name": line}):
                GNew.remove_node(line)
        GNew.remove_nodes_from(list(nx.isolates(GNew)))
    # handle degree input
    if degree != '1':
        nodesremove = [node1 for node1 in GNew.nodes() if GNew.degree(node1) < int(degree)]
        print(nodesremove)
        GNew.remove_nodes_from(nodesremove)
        #         # GNew.remove_node(node1)

def update():
    GNew = G.copy()
    degree = degree_select.value
    network = network_multiseclect.value
    dropNode(GNew, network, degree)
    node_size = list(v['node_size'] for k, v in GNew.nodes(data=True))
    newplot = figure(title="FB network", tools="tap", width=1500, height=800)
    newgraph = from_networkx(GNew, nx.spring_layout(GNew), scale=1, center=(0, 0))
    newplot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())
    url = "@linkUser"
    taptool = newplot.select(type=TapTool)
    taptool.callback = OpenURL(url=url)
    newgraph.node_renderer.data_source.data['node_size'] = node_size
    newgraph.node_renderer.glyph = Circle(size='node_size')
    newplot.renderers.append(newgraph)

    # add label to node.
    x, y = zip(*newgraph.layout_provider.graph_layout.values())
    node_labels = list(GNew.nodes())

    source = ColumnDataSource({'x': x, 'y': y, 'index': [node_labels[i] for i in range(len(x))]})

    labels = LabelSet(x='x', y='y', text='index', source=source,
                      background_fill_color='white')
    newplot.renderers.append(labels)

    layout.children[1] = newplot

    # draw_tool = PointDrawTool(renderers=[g.node_renderer], empty_value='black')
    # p.add_tools(draw_tool)
    # p.toolbar.active_tap = draw_tool22

def updateGraph(attr, old, new):
    update()

# def updatePlot(attrname, old, new):
#     degree = new
#     # xoa tat ca cac node < new
#     # ve lai plot
#     GNew = G.copy()
#     dropNode(GNew, new, type='degree')
#     node_size = list(v['node_size'] for k, v in GNew.nodes(data=True))
#     newplot = figure(title="FB network", tools="tap", width=1500, height=800)
#     newgraph = from_networkx(GNew, nx.spring_layout(GNew), scale=1, center=(0, 0))
#     newplot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())
#     url = "@linkUser"
#     taptool = newplot.select(type=TapTool)
#     taptool.callback = OpenURL(url=url)
#     newgraph.node_renderer.data_source.data['node_size'] = node_size
#     newgraph.node_renderer.glyph = Circle(size='node_size')
#     newplot.renderers.append(newgraph)
#
#     # add label to node.
#     x, y = zip(*newgraph.layout_provider.graph_layout.values())
#     node_labels = list(GNew.nodes())
#
#     source = ColumnDataSource({'x': x, 'y': y, 'index': [node_labels[i] for i in range(len(x))]})
#     print('node_new')
#     print(node_labels)
#     print(x)
#     print(len(x))
#
#     labels = LabelSet(x='x', y='y', text='index', source=source,
#                       background_fill_color='white')
#     newplot.renderers.append(labels)
#
#     layout.children[0] = newplot


#  khi thay đổi lại cấu trúc thì sẽ có những thay đổi sau:
# biểu đồ thay đổi
# các trường trong multiselect cũng thay đổi theo.
def updatePlotWhenChange(attr, old, new):

    number_post = int(number_post_select.value)
    number_post_connection = int(number_post_connection_select.value)
    dataPost = data1[data1.TYPE == 'POST'][0:number_post]
    global listPost
    listPost = list(dataPost['PAGE NAME'])
    print(listPost)

    dataComment = pd.DataFrame()
    for index, row in dataPost.iterrows():
        # SỬA: Lấy ra number_post_con comment nhiều tương tác nhất tương ứng vs các bài Post
        dataComment = dataComment.append(
            dataRoot.loc[(dataRoot['PAGE NAME'] == row['USER NAME']) & (dataRoot['TYPE'] == 'COMMENT')][0:number_post_connection])
        # if row['USER NAME'] == row['PAGE NAME']:
        #     dataPost.at[index, 'TTT'] = row['TTT'] / 2
    data = dataPost.append(dataComment)

    # make link_url for tap nodes
    linkUser = {}
    for index, row in data.iterrows():
        linkUser[row['PAGE NAME']] = 'https://facebook.com/' + str(row['PAGE ID'])
        linkUser[row['USER NAME']] = 'https://facebook.com/' + str(row['USER ID'])
    global G
    G = nx.from_pandas_edgelist(data,
                                source='USER NAME',
                                target='PAGE NAME',
                                edge_attr='TTT',
                                create_using=nx.DiGraph())
    nx.set_node_attributes(G, linkUser, "linkUser")

    weighted_degrees = dict(G.in_degree(weight='TTT'))

    pos = nx.spring_layout(G, k=0.5, iterations=60, scale=10)
    max_weight = max(weighted_degrees, key=lambda x: weighted_degrees[x])
    node_size = []

    for node in G.nodes():
        a = weighted_degrees[node] * 100 / weighted_degrees[max_weight]
        node_size.append(max(a, 10))
        G.nodes[node]['node_size'] = max(a, 10)

    network_multiseclect.options = [(i, i) for i in listPost]
    # network_multiseclect = MultiSelect(title='An cac bai viet', options=[(i, i) for i in listPost], value=[''])

    update()
    # newplot = figure(title="FB network", tools="tap", width=1500, height=800)
    # newgraph = from_networkx(G, nx.spring_layout(G), scale=1, center=(0, 0))
    # newplot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())
    # url = "@linkUser"
    # taptool = newplot.select(type=TapTool)
    # taptool.callback = OpenURL(url=url)
    # newgraph.node_renderer.data_source.data['node_size'] = node_size
    # newgraph.node_renderer.glyph = Circle(size='node_size')
    # newplot.renderers.append(newgraph)
    #
    # # add label to node.
    # x, y = zip(*newgraph.layout_provider.graph_layout.values())
    # node_labels = list(G.nodes())
    #
    # source = ColumnDataSource({'x': x, 'y': y, 'index': [node_labels[i] for i in range(len(x))]})
    #
    # labels = LabelSet(x='x', y='y', text='index', source=source,
    #                   background_fill_color='white')
    # newplot.renderers.append(labels)
    #
    # layout.children[1] = newplot


network_multiseclect.on_change('value', updateGraph)

g = from_networkx(G, pos, scale=1, center=(0, 0))

node_hover_tool = HoverTool(tooltips=[("name", "@index"), ("linkFB", "@linkUser")])
p.add_tools(node_hover_tool, BoxZoomTool(), ResetTool(), PanTool(), PolySelectTool(), WheelZoomTool(), SaveTool())

# Configure tap tool
url = "@linkUser"
taptool = p.select(type=TapTool)
taptool.callback = OpenURL(url=url)

# draw_tool = PointDrawTool(renderers=[g.node_renderer], empty_value='black')
# p.add_tools(draw_tool)
# p.toolbar.active_tap = draw_tool

g.node_renderer.data_source.data['node_size'] = node_size
g.node_renderer.glyph = Circle(size='node_size')

# g.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
# g.edge_renderer.glyph = MultiLine(line_alpha=0.7, line_width=1)
p.renderers.append(g)

# add label to node.
x, y = zip(*g.layout_provider.graph_layout.values())
node_labels = list(G.nodes())
source = ColumnDataSource({'x': x, 'y': y,
                           'index': [node_labels[i] for i in range(len(x))]})
labels = LabelSet(x='x', y='y', text='index', source=source,
                  background_fill_color='white')
p.renderers.append(labels)

# Hiển thị số liên kết tối thiểu
degree_options = ['1', '2', '3', '4']
degree_select = Select(value='1', title="Số liên kết tối thiểu:", options=degree_options)
degree_select.on_change('value', updateGraph)


# Giới hạn số bài post hiển thị
number_post_options = ['5', '7', '10']
number_post_select = Select(value='5', title="Giới hạn số mạng con hiển thị :", options=number_post_options)
number_post_select.on_change('value', updatePlotWhenChange)

# Giới hạn số liên kết với mỗi bài post:
number_post_connection_options = ['4', '5', '6', '7']
number_post_connection_select = Select(value='5', title="Giới hạn số liên kết với mỗi bài viết :", options=number_post_connection_options)
number_post_connection_select.on_change('value', updatePlotWhenChange)

custom_option = Column(number_post_select, number_post_connection_select)
rows = row(degree_select, network_multiseclect, custom_option)
layout = Column(rows, p)
curdoc().add_root(layout)
