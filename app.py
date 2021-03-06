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
        {'label': '高卒未満', 'value': '1'},
        {'label': '高卒', 'value': '2'},
        {'label': '大学中退', 'value': '3'},
        {'label': '大卒', 'value': '4'}
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
        {'label': '管理職、専門職', 'value': '1'},
        {'label': '技術職、営業職、サービス業', 'value': '2'},
        {'label': 'その他', 'value': '3'},
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
    html.Br(),
    html.Br(),
    html.Button(id="submit-button", children="判定"),
    dcc.Graph(id="output-state"),
    dcc.Graph(id="epl-graph"),
    dcc.Graph(id="pfm-graph"),
    dcc.Graph(id="sim-graph")
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
    riskreturns = data_import('epl.csv')
    wi = [0,10,20,30,39]
    port_riskseturns = riskreturns.iloc[wi]
    asset_riskseturns = data_import('asrkrt.csv')
    
    
    # グラフの記述
    fig = go.Figure(layout=go.Layout(
                title = 'リスク・リターン',
                height = 400, 
                width = 800,
                xaxis = dict(title="リスク",range=[0, 0.08]),
                yaxis = dict(title="リターン",range=[-0.02, 0.08])
            )
        )

    

    # 効率的フロンティア
    fig.add_traces(go.Scatter(
        x = riskreturns['risks'],
        y = riskreturns['returns'],
        mode='lines+markers',
        name='効率的フロンティア',
        marker_color='rgba(255, 182, 193, .9)'
    ))

    # 各資産のリスクリターン
    fig.add_traces(go.Scatter(
        x = asset_riskseturns['risks'],
        y = asset_riskseturns['returns'],
        text = list(asset_riskseturns.index),
        textposition='middle right',
        mode='markers+text',
        name='構成資産',
        showlegend=False
    ))

    # 提案ポートフォリオのリスクリターン
    fig.add_traces(go.Scatter(
        x = pd.Series(port_riskseturns['risks'].iloc[RiskTolerance]),
        y = pd.Series(port_riskseturns['returns'].iloc[RiskTolerance]),
        mode='markers',
        name='提案ポートフォリオ',
        marker = dict(size=15),
        marker_symbol = 'star',
        marker_color='rgba(152, 0, 0, .8)'
    ))

    
    return fig


@app.callback(
    Output("pfm-graph", "figure"),
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
def pfm_graph(n_clicks,Age,Edu,Married,Kids,Occ,Inccl,Nwcat,Risk):
    X_input = [[int(Age),int(Edu),int(Married),int(Kids),int(Occ),int(Inccl), int(Risk),int(Nwcat)]]
    RiskTolerance = predict_riskTolerance(X_input)
    portsets = ['defensive','slightly-defensive','middle','slightly-aggressive','aggressive']
    file = 'portperformance_' + portsets[int(RiskTolerance)-1] + '.csv'
    pfm_data = data_import(file)
    
    # グラフの記述
    fig = go.Figure(layout=go.Layout(
                title = '提案ポートフォリオのパフォーマンス(2018年1月に100万円投資した場合)',
                height = 400, 
                width = 800,
                xaxis = dict(title="年月"),
                yaxis = dict(title="評価額")
            )
        )

    # パフォーマンス
    fig.add_traces(go.Scatter(
        x = pfm_data['日付'],
        y = pfm_data['ポートフォリオ'],
        mode='lines',
        name='提案ポートフォリオ',
        showlegend=True
        # marker_color='rgba(255, 182, 193, .9)'
    ))
    
    return fig


@app.callback(
    Output("sim-graph", "figure"),
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
def sim_graph(n_clicks,Age,Edu,Married,Kids,Occ,Inccl,Nwcat,Risk):
    X_input = [[int(Age),int(Edu),int(Married),int(Kids),int(Occ),int(Inccl), int(Risk),int(Nwcat)]]
    RiskTolerance = predict_riskTolerance(X_input)
    riskreturns = data_import('epl.csv')
    wi = [0,10,20,30,39]
    port_riskseturns = riskreturns.iloc[wi]
    mu = port_riskseturns['returns'].iloc[int(RiskTolerance)]
    sigma = port_riskseturns['risks'].iloc[int(RiskTolerance)]
    ganpon = 1000000
    year = 30
    rappaData = create_rappaData(ganpon,year,mu,sigma)
    ri = ['m3','m2','m1','c','p1','p2','p3']
    # rappaData.to_csv('rd.csv')
    
    # グラフの記述
    fig = go.Figure(layout=go.Layout(
                title = '提案ポートフォリオの将来シミュレーション',
                height = 400, 
                width = 800,
                xaxis = dict(title="年月"),
                yaxis = dict(title="予想額")
            )
        )

    # +2σ
    fig.add_traces(go.Scatter(
        x = rappaData['year'],
        y = rappaData['p2'],
        mode='lines',
        name='+2σ',
        marker_color='rgba(229,153,255, .9)'
    ))
    
    # +1σ
    fig.add_traces(go.Scatter(
        x = rappaData['year'],
        y = rappaData['p1'],
        mode='lines',
        name='+1σ',
        marker_color='rgba(204,50,255, .9)'
    ))

    # 中央値
    fig.add_traces(go.Scatter(
        x = rappaData['year'],
        y = rappaData['c'],
        mode='lines',
        name='中央値',
        marker_color='rgba(152,0,203, .9)'
    ))

    # -1σ
    fig.add_traces(go.Scatter(
        x = rappaData['year'],
        y = rappaData['m1'],
        mode='lines',
        name='-1σ',
        marker_color='rgba(204,50,255, .9)'
    ))

    # -2σ
    fig.add_traces(go.Scatter(
        x = rappaData['year'],
        y = rappaData['m2'],
        mode='lines',
        name='-2σ',
        marker_color='rgba(229,153,255, .9)'
    ))
    
    return fig


def data_import(filename):
    df = pd.read_csv(filename, index_col=0)
    return df


def predict_riskTolerance(X_input):
    filename = 'finalized_model_xgbclass.sav'
    loaded_model = load(open(filename, 'rb'))
    X_input = pd.DataFrame(X_input,columns=['AGE07', 'EDCL07', 'MARRIED07', 'KIDS07', 'OCCAT107', 'INCOME07','RISK07', 'NETWORTH07'])
    predictions = loaded_model.predict(X_input)
    return predictions


def calc_weight(riskTolerance):
    portweightcsv = pd.read_csv('portweight.csv', index_col=0) 
    w = portweightcsv.iloc[riskTolerance-1]
    w = w.values.flatten().tolist()
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


def create_rappaData(ganpon,year,mu,sigma):
    # 連続福利収益率の平均標準偏差に変換
    s = np.sqrt(np.log(1+(sigma/(1+mu))**2))
    r = np.log(1+mu)-(s**2)/2
    # ラッパデータ作成
    rappa = pd.DataFrame(np.arange(year+1),columns=['year'])
    rappa['mean'] = rappa['year']*r
    rappa['sd'] = np.sqrt(rappa['year'])*s
    rappa['m3'] = ganpon*np.exp(rappa['mean']-3*rappa['sd'])
    rappa['m2'] = ganpon*np.exp(rappa['mean']-2*rappa['sd'])
    rappa['m1'] = ganpon*np.exp(rappa['mean']-1*rappa['sd'])
    rappa['c'] = ganpon*np.exp(rappa['mean']-0*rappa['sd'])
    rappa['p1'] = ganpon*np.exp(rappa['mean']+1*rappa['sd'])
    rappa['p2'] = ganpon*np.exp(rappa['mean']+2*rappa['sd'])
    rappa['p3'] = ganpon*np.exp(rappa['mean']+3*rappa['sd'])
    return rappa


if __name__ == '__main__':
    app.run_server(debug=True)