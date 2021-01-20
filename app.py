import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from pickle import load
import numpy as np
import pandas as pd
import cvxopt as opt
import plotly.graph_objects as go
from numpy import array

app = dash.Dash()

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
    dcc.Graph(id="output-state")
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
    X_input = [[Age,Edu,Married,Kids,Occ,Inccl, Risk,Nwcat]]
    RiskTolerance = predict_riskTolerance(X_input)
    weight = calc_weight(RiskTolerance)
    # return create_pieChart(weight)
    # グラフの記述
    figure = {
        'data': [
            go.Pie(
                labels = ['国内株式','国内債券','国内リート','先進国株式','新興国株式','先進国債券','新興国債券','先進国リート'],
                values = weight,
                # values = [1,2,3,4,5,6,7,8],
                name='train data'
            )
        ],
        'layout':{
            'title': '提案ポートフォリオ'
        }
    }
    return figure


def predict_riskTolerance(X_input):
    filename = 'finalized_model.sav'
    loaded_model = load(open(filename, 'rb'))
    predictions = loaded_model.predict(X_input)
    return predictions


def calc_weight(riskTolerance):
    # リスクリターンを読み込み
    returns = pd.read_csv('returns.csv', index_col=0) 
    returns = np.power(returns+1,1/3)-1
    covr = pd.read_csv('covr.csv', index_col=0) * np.sqrt(250)

    # 最適ポートフォリオ算出
    n = len(returns)
    min_ratio = 0.03
    max_ratio = 0.5
    min_r = 0.02
    mus = 1-riskTolerance
    P = mus*covr.values
    P_m = opt.matrix(P)
    q = np.zeros((n, 1))
    q_m = opt.matrix(q)
    G = np.concatenate((-np.identity(n), np.identity(n)), 0)
    G = np.concatenate((G,-np.transpose(returns.values)),0)
    G_m = opt.matrix(G)
    h = np.concatenate((-np.ones((n,1))*min_ratio, np.ones((n,1))*max_ratio), 0)
    h = np.concatenate((h,-np.ones((1,1))*min_r),0)
    h_m = opt.matrix(h)
    A = np.ones((1,n))
    A_m = opt.matrix(A)
    b = np.ones((1,1))
    b_m = opt.matrix(b)

    portfolios = opt.solvers.qp(P_m, q_m, G_m, h_m, A_m, b_m)
    weight = portfolios['x']
    w = array(weight)
    w = w.flatten()
    w = w.tolist()
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