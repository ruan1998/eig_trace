import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_daq as daq
from flask_caching import Cache
import plotly.graph_objs as go
from datetime import datetime
import pandas as pd
import numpy as np
from utils.get_eig_values import Eigenval
import os
import time

day_time = 10
click_num = 0
run_times = 0
df = pd.read_csv('data/feature_points.csv')
type_nums = len(df.groupby('IM_type'))
trace_types = range(type_nums)

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash()
# CACHE_CONFIG = {
#     # try 'filesystem' if you don't want to setup redis
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR':'cache'
# }
# cache = Cache()
# cache.init_app(app.server, config=CACHE_CONFIG)

to_run_df = pd.DataFrame(index=range(day_time*type_nums), columns=['real','imag','time'])



app.layout = html.Div([
    html.Div(id='plot_signal', style={'display': 'none'}),
    html.Div([
        html.Div([
            html.H1('特征值追踪交互Web应用', style={'text-align':'right'})
            ],style={'width':'65%','display':'inline-block'}),
        html.Div([
            daq.PowerButton(id='start_button',color='#FF5E5E',on=False,
                            label='运行程序',labelPosition='top')
            ],style={'width':'33%','display':'inline-block'})
            ]),
    html.H3("选择特征值追踪方法："),
    html.Div([
    html.Div([
    dcc.Dropdown(
        id='methods',
        options=[
            {'label': '启发式算法', 'value': 'IM'},
            {'label': '随机森林', 'value': 'RF'},
            {'label': '模拟退火', 'value': 'SA'}
        ],
        value='IM'
    )],style={'width':'48%','display':'inline-block'}),
    html.Div([dcc.Input(placeholder='输入轨迹标签...',id='trace', value='',type='', style={'fontSize':25}),
    html.Button('Submit', id='button',n_clicks=0, style={'fontSize':25})
    ],style={'width':'48%', 'display':'inline-block','vertical-align':'top'})
    ]),
    dcc.Graph(id='extracite_points',figure={'layout': {
                'clickmode': 'event+select'
            }}),
    dcc.RangeSlider(
        id='time_range',
        marks={i:f"{'' if i%2!=0 else i}" for i in range(96)},
        min=0,
        max=95,
        value=[0, 95])
])

# @cache.memoize()
# def global_store(is_run):
#     global run_times, type_nums, day_time, to_run_df
#     # simulate expensive query
#     print(f'[computing]:{run_times}')
#     result = Eigenval(run_times)
#     to_run_df.loc[run_times*type_nums:(run_times+1)*type_nums-1,'real'] = result.real
#     to_run_df.loc[run_times*type_nums:(run_times+1)*type_nums-1,'imag'] = result.imag
#     to_run_df.loc[run_times*type_nums:(run_times+1)*type_nums-1,'time'] = [run_times]*type_nums
#     run_times += 1
#     return True if run_times < day_time else False

@app.callback(Output('plot_signal', 'children'), [Input('start_button', 'on')])
def compute_value(is_run):
    global run_times, type_nums, day_time, to_run_df
    if is_run and run_times < day_time:
        print(f'[computing]:{run_times}')
        result = Eigenval(run_times)
        to_run_df.loc[run_times*type_nums:(run_times+1)*type_nums-1,'real'] = result.real
        to_run_df.loc[run_times*type_nums:(run_times+1)*type_nums-1,'imag'] = result.imag
        to_run_df.loc[run_times*type_nums:(run_times+1)*type_nums-1,'time'] = [run_times]*type_nums
        run_times += 1
        # if not to_plot:
        #     raise dash.exceptions.PreventUpdate
        return True
    elif run_times == day_time:
        run_times += 1
        return True
    else:
        raise dash.exceptions.PreventUpdate

@app.callback([Output('extracite_points', 'figure'),
                Output('start_button', 'on')],
              [Input('plot_signal', 'children'),
               Input('methods', 'value'),
               Input('time_range', 'value'),
               Input('button', 'n_clicks')],
               [State('trace', 'value')]
               )
def update_graph_method(to_plot, method, t_range, n_clicks, trace_num):
    global click_num , trace_types, type_nums, run_times, day_time, to_run_df
    if to_plot and run_times <= day_time:
        button_on = True if run_times < day_time else False
        return {'data': [go.Scatter(x=to_run_df.real[:run_times*type_nums],
                                    y=to_run_df.imag[:run_times*type_nums],
                                    text=to_run_df.time[:run_times*type_nums],
                                    mode='markers',
                                    marker={'size': 6,
                                            # 'color': np.linspace(0, 1,1+t_range[1]-t_range[0]),
                                            # 'colorscale':'hot',
                                            'opacity': 0.2,
                                            'line': {'width': 0.3, 'color': 'black'}}
                                    )
                        for i in range(run_times)],

                'layout': go.Layout(title=f'{method}特征值追踪',
                                    xaxis={'title': 'real'},
                                    yaxis={'title': 'imag'},
                                    hovermode='closest'),
            }, button_on 
    # elif run_times < day_time:
    #     raise dash.exceptions.PreventUpdate
    # elif run_times < day_time - 1:
    #     return {'data':None, 'layout':None}, False
    else:
        if n_clicks > click_num:
            if '-' in trace_num:
                start, end = trace_num.split('-')
                trace_types = range(int(start), int(end)+1)
                click_num += 1
            else:
                trace_types = [int(i) for i in trace_num.split(' ')]
                click_num += 1
        now_df = df[(df.time >= t_range[0]) & (df.time <= t_range[1])]
        return {'data': [go.Scatter(x=now_df[now_df[f'{method}_type'] == i]['real'],
                                    y=now_df[now_df[f'{method}_type'] == i]['imag'],
                                    text=now_df[now_df[f'{method}_type'] == i]['time'],
                                    mode='lines+markers',
                                    name=f'Trace {i}',
                                    line={'width':1,
                                          # 'color':np.linspace(0, 1,1+t_range[1]-t_range[0]),
                                          # 'dash':'longdash'},
                                          },
                                    marker={'size': 6,
                                            'color': np.linspace(0, 1,1+t_range[1]-t_range[0]),
                                            'colorscale':'hot',
                                            'opacity': 0.5,
                                            'line': {'width': 0.5, 'color': 'black'}}
                                    )
                         for i in trace_types],

                'layout': go.Layout(title=f'{method}特征值追踪',
                                    xaxis={'title': 'real'},
                                    yaxis={'title': 'imag'},
                                    hovermode='closest'),
                }, None


if __name__ == '__main__':
    app.run_server(host='0.0.0.0')

 # A list of colors that will be spaced evenly to create the colorscale.
 #        Many predefined colorscale lists are included in the sequential, diverging,
 #        and cyclical modules in the plotly.colors package.
 #      - A list of 2-element lists where the first element is the
 #        normalized color level value (starting at 0 and ending at 1), 
 #        and the second item is a valid color string.
 #        (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])
 #      - One of the following named colorscales:
 #            ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
 #             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
 #             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
 #             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
 #             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
 #             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
 #             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
 #             'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
 #             'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor',
 #             'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy',
 #             'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral',
 #             'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose',
 #             'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'twilight',
 #             'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd']