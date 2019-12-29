import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from datetime import datetime
import pandas as pd
import numpy as np

click_num = 0
df = pd.read_csv('data/feature_points.csv')
type_nums = len(df.groupby('IM_type'))
trace_types = range(type_nums)

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app = dash.Dash()

app.layout = html.Div([
    html.H1('特征值追踪交互Web应用', style={'text-align':'center'}),
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

@app.callback(Output('extracite_points', 'figure'),
              [Input('methods', 'value'),
               Input('time_range', 'value'),
               Input('button', 'n_clicks')],
               [State('trace', 'value')
               ])
def update_graph_method(method, t_range, n_clicks, trace_num):
    global click_num , trace_types
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
            }


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