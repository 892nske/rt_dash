import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from pickle import load
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import array
import os

app = dash.Dash()

server = app.server

app.layout = html.Div([
    html.Label('年齢'),
    html.Br(),
    dcc.Input(
        id="input-age",
        type="text", 
        value="30",
    ),
    html.Br(),
    html.Br(),
    html.Label('学歴'),
    dcc.RadioItems(
        id="radio-edu",
        options=[
        {'label': '高卒', 'value': '1'},
        {'label': '学部卒', 'value': '2'},
        {'label': '修士卒', 'value': '3'},
        {'label': '博士卒', 'value': '4'}
        ],
        value='1',
        labelStyle={'display': 'inline-block'}
    ),
    html.Br(),
    html.Label('結婚'),
    dcc.RadioItems(
        id="radio-mar",
        options=[
        {'label': '既婚', 'value': '1'},
        {'label': '未婚', 'value': '2'}
        ],
        value='1',
        labelStyle={'display': 'inline-block'}
    ),
    html.Br(),
    html.Label('子供の数'),
    html.Br(),
    dcc.Input(
        id="input-kid",
        type="text", 
        value="0",
    ),
    html.Br(),
    html.Br(),
    html.Label('職業'),
    dcc.RadioItems(
        id="radio-occ",
        options=[
        {'label': '仕事１', 'value': '1'},
        {'label': '仕事２', 'value': '2'},
        {'label': '仕事３', 'value': '3'},
        {'label': '無職', 'value': '4'}
        ],
        value='1',
        labelStyle={'display': 'inline-block'}
    ),
    html.Br(),
    html.Label('収入'),
    html.Br(),
    dcc.Input(
        id="input-inc",
        type="text", 
        value="0",
    ),
    html.Br(),
    html.Br(),
    html.Label('保有資産'),
    html.Br(),
    dcc.Input(
        id="input-ntw",
        type="text", 
        value="0",
    ),
    html.Br(),
    html.Br(),
    html.Label('リスク志向'),
    dcc.RadioItems(
        id="radio-rsk",
        options=[
        {'label': '積極', 'value': '1'},
        {'label': 'やや積極', 'value': '2'},
        {'label': 'やや安定', 'value': '3'},
        {'label': '安定', 'value': '4'}
        ],
        value='1',
        labelStyle={'display': 'inline-block'}
    ),
    html.Button(id="submit-button", children="判定"),
    dcc.Graph(id="output-state"),
    dcc.Graph(id="epl-graph")
])

@app.callback(
    Output("output-state", "figure"),
    [Input("submit-button", 'n_clicks')],
    [State("input-age", "value"),
     State("radio-edu", "value"),
     State("radio-mar", "value"),
     State("input-kid", "value"),
     State("radio-occ", "value"),
     State("input-inc", "value"),
     State("input-ntw", "value"),
     State("radio-rsk", "value")]
)
def update_output(n_clicks,Age,Edu,Married,Kids,Occ,Inccl,Nwcat,Risk):
    X_input = [[int(Age),int(Edu),int(Married),int(Kids),int(Occ),int(Inccl), int(Risk),int(Nwcat)]]
    RiskTolerance = predict_riskTolerance(X_input)
    weight = calc_weight(RiskTolerance)
    # グラフの記述
    figure = {
        'data': [
            go.Pie(
                labels = ['国内株式','国内債券','国内リート','先進国株式','新興国株式','先進国債券','新興国債券','先進国リート'],
                values = weight,
                name='train data'
            )
        ],
        'layout':{
            'title': '提案ポートフォリオ',
            'width': '500'
        }
    }
    return figure


@app.callback(
    Output("epl-graph", "figure"),
    [Input("submit-button", 'n_clicks')],
    [State("input-age", "value"),
     State("radio-edu", "value"),
     State("radio-mar", "value"),
     State("input-kid", "value"),
     State("radio-occ", "value"),
     State("input-inc", "value"),
     State("input-ntw", "value"),
     State("radio-rsk", "value")]
)
def efficient_portfolio(n_clicks,Age,Edu,Married,Kids,Occ,Inccl,Nwcat,Risk):
    X_input = [[int(Age),int(Edu),int(Married),int(Kids),int(Occ),int(Inccl), int(Risk),int(Nwcat)]]
    RiskTolerance = predict_riskTolerance(X_input)
    riskreturns = epl_riskreturn()
    # グラフの記述
    trace0 = go.Scatter(
                x = riskreturns['risks'],
                y = riskreturns['returns'],
                mode='markers',
                name='効率的フロンティア',
            )
    trace1 = go.Scatter(
                x = pd.DataFrame([0.02,0.05]),
                y = pd.DataFrame([0.05,0.06]),
                mode='markers',
                name='提案ポートフォリオ',
            )

    # figure = {
    #     'data': [
    #         trace0,trace1
    #         # go.Scatter(
    #         #     x = riskreturns['risks'],
    #         #     y = riskreturns['returns'],
    #         #     name='効率的フロンティア',
    #         # ),
    #         # go.Scatter(
    #         #     x = pd.DataFrame([0.5]),
    #         #     y = pd.DataFrame([0.5]),
    #         #     name='提案ポートフォリオ',
    #         # )
    #     ],
    #     'layout':{
    #         'title': 'リスク・リターン',
    #         'width': '800',
    #         'xaxis': dict(
    #                 title="リスク",
    #                 range=[0, 0.08],
    #         ),
    #         'yaxis': dict(
    #             title="リターン",
    #             range=[0, 0.06]
    #         )
    #     }
    # }
    t = np.linspace(0, 10, 100)

    fig = go.Figure()

    

    fig.add_trace(go.Scatter(
        x = pd.DataFrame([0.03,0.03]),
        y = pd.DataFrame([0.03,0.03]),
        mode='markers',
        name='提案ポートフォリオ',
        marker_color='rgba(255, 182, 193, .9)'
    ))

    fig.add_trace(go.Scatter(
        x = riskreturns['risks'],
        y = riskreturns['returns'],
        mode='lines+markers',
        name='効率的フロンティア',
        marker_color='rgba(152, 0, 0, .8)'
    ))
    
    return fig
    
def epl_riskreturn():
    eplcsv = pd.read_csv('epl.csv', index_col=0)
    return eplcsv


def predict_riskTolerance(X_input):
    filename = 'finalized_model_xgbclass.sav'
    loaded_model = load(open(filename, 'rb'))
    X_input = pd.DataFrame(X_input,columns=['AGE07', 'EDCL07', 'MARRIED07', 'KIDS07', 'OCCAT107', 'INCOME07','RISK07', 'NETWORTH07'])
    predictions = loaded_model.predict(X_input)
    return predictions


def calc_weight(riskTolerance):
    portweightcsv = pd.read_csv('portweight.csv', index_col=0) 
    w = portweightcsv.iloc[riskTolerance-1]
    # w = portweightcsv.iloc[0]
    w = w.values.flatten().tolist()
    # w = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.3]
    return w

def create_pieChart(weight):
    return {
        'data':[go.Pie(
            labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen'],
            values = [4500, 2500, 1053, 500]
        )],
        'layout':{
            'title': '提案ポートフォリオ'
        }
    }

if __name__ == '__main__':
    app.run_server(debug=True)