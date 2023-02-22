# -*- coding: utf-8 -*-

import networkx as nx

from bokeh.plotting import figure, from_networkx, curdoc, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool,
                          ColumnDataSource, OpenURL, TapTool, PointDrawTool, Rect, TextInput)
from bokeh.layouts import column

global graph
G = nx.Graph()

field = TextInput(value="first", title="Graph to select: ")


def graphe(base, H):
    for i in range(0, 10):
        H.add_node(base + str(i), name=base + str(i),
                   version=str(i),
                   width=.2,
                   offset=- 25,
                   color=' red')
        H.add_edge(base + str(0), base + str(i))

def dropNode(base,H):
    G.remove_node(base + '1')

def update(attr, old, new):
    # newG = nx.Graph()
    # graphe(new, newG)
    print(new)
    print(old)
    dropNode(old, G)
    newplot = figure(title="RPM network", width=1500, height=800, x_range=(-2.1, 2.1), y_range=(-2.1, 2.1),
                     tools="", toolbar_location=None)
    newgraph = from_networkx(G, pos, scale=1, center=(0, 0))
    newgraph.node_renderer.glyph = Rect(height=0.1, width="width", fill_color="color")
    newplot.add_tools(hover, BoxZoomTool(), ResetTool())
    newplot.renderers.append(newgraph)
    layout.children[1] = newplot


graphe("first", G)
pos = nx.spring_layout(G)
nx.draw_networkx(G)
plot = figure(title="RPM network", width=1500, height=800, x_range=(-2.1, 2.1), y_range=(-2.1, 2.1), toolbar_location=None)

hover = HoverTool()
hover.tooltips = """
<div style=padding=5px>@name</div>
<div style=padding=5px>@version</div>
"""
plot.add_tools(hover, BoxZoomTool(), ResetTool())
graph = from_networkx(G, pos, center=(0, 0))
graph.node_renderer.glyph = Rect(height=0.1, width="width", fill_color="color")
plot.renderers.append(graph)
field.on_change('value', update)

layout = column(field, plot)
curdoc().add_root(layout)
show(layout)