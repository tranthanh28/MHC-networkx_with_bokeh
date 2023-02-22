from bokeh.models import ColumnDataSource, MultiSelect, Column
from bokeh.plotting import figure, curdoc, show
from datetime import datetime
from random import randint
from bokeh.palettes import Category10

lines = ['line_{}'.format(i) for i in range(10)]
data = [{'time': [], item: []} for item in lines]
sources = [ColumnDataSource(item) for item in data]

plot = figure(plot_width=1200, x_axis_type='datetime')


def add_line(attr, old, new):
    for line in new:
        if not plot.select_one({"name": line}):
            index = lines.index(line)
            plot.line(x='time', y=line, color=Category10[10][index], name=line, source=sources[index])


multiselect = MultiSelect(title='Options', options=[(i, i) for i in lines], value=[''])
multiselect.on_change('value', add_line)


def update():
    for line in lines:
        if line in multiselect.value:
            if plot.select({"name": line}):
                sources[lines.index(line)].stream(
                    eval('dict(time = [datetime.now()], ' + line + ' = [randint(5, 10)])'))


curdoc().add_root(Column(plot, multiselect))
curdoc().add_periodic_callback(update, 1000)
