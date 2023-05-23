import pandas as pd
import numpy as np

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import plotly.express as px

from approximative_weak_solutions import plot_solution


"""
Define app, and setup components and layout of app.
"""

app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])

app.layout = dbc.Container([

    dcc.ConfirmDialog(id='error_message', message='', displayed=False),

    dbc.Row(
        html.H1('Approximative Solutions to Differential Equations', className='text-center')
    ),

    dbc.Row(
        html.Hr()
    ),

    dbc.Row([

        dbc.Col([

            html.H4('Differential Equation'),
            dcc.Dropdown(id='differential_equation', value='Poisson',
                         options=['Poisson', 'Sturm-Liouville']
            ),

            html.Br(),
            html.H4('Method'),
            dcc.Dropdown(id='method', value='Eigenfunction Decomposition',
                         options=['Eigenfunction Decomposition', 'Piecewise Linear']
            ),

            html.Br(),
            html.H4('Input Function'),
            dcc.Textarea(id='input_function', value='0'),

            html.Br(),
            dcc.Slider(id='complexity', value=5, min=1, max=20, step=1),

            html.Br(),
            dbc.Button(id='draw_button', children='Approximate!', n_clicks=0)

        ], width={'size':2, 'order':1}
        ),

        dbc.Col(
            dcc.Graph(id='approximation_graph', figure={}),
            width={'size':8, 'offset':1, 'order':2}
        )

    ], align='center'
    ),
], fluid=True)


"""
Callback to compute approximative solution and display it.
"""

@app.callback(
    Output('approximation_graph', 'figure'),
    Output('error_message', 'message'),
    Output('error_message', 'displayed'),
    Input('draw_button', 'n_clicks'),
    State('differential_equation', 'value'),
    State('method', 'value'),
    State('input_function', 'value'),
    State('complexity', 'value')
)
def graph(clicks, diff_eq, method, h_text, n):
    change = {t:'np.'+t for t in ['sqrt', 'pi', 'exp', 'sin', 'cos', 'tan']}
    change['^'] = '**'
    for c in change.keys():
        h_text = h_text.replace(c, change[c])
    h = lambda x: eval(h_text)

    method = method.replace('Eigenfunction Decomposition', 'eigenfunction decomposition')
    method = method.replace('Piecewise Linear', 'piecewise linear')

    try:
        if diff_eq == 'Poisson':
            solution = plot_solution(diff_eq, method, h, int(n))
            return px.line(x=np.linspace(0, np.pi, 100), y=solution), '', False

        if diff_eq == 'Sturm-Liouville':
            solution = plot_solution(diff_eq, method, h, int(n))
            return px.line(x=np.linspace(1, 2, 100), y=solution), '', False

    except Exception as error:
        if diff_eq == 'Poisson':
            return px.line(x=np.linspace(0, np.pi, 100), y=np.repeat(0, 100)), 'Error: ' + str(error), True

        if diff_eq == 'Sturm-Liouville':
            return px.line(x=np.linspace(1, 2, 100), y=np.repeat(0, 100)), 'Error: ' + str(error), True


if __name__ == '__main__':
    app.run_server(debug=True, port=3000)