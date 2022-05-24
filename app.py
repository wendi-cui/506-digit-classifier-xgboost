import numpy as np
from skimage import io
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import html, dcc
from dash_canvas import DashCanvas
from dash_canvas.utils import  parse_jsonstring
# from dash_canvas.utils import array_to_data_url
# from PIL import Image
# from io import BytesIO
# import base64
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import pickle


########### open the pickle file ######
filename = open('analysis/tree_model.pkl', 'rb')
tree_model = pickle.load(filename)
filename.close()

########### define variables
tabtitle='digits classifier'
sourceurl = 'https://www.kaggle.com/burak3ergun/loan-data-set'
githublink = 'https://github.com/plotly-dash-apps/504-mortgage-loans-predictor'
canvas_size = 200

########### BLANK FIGURE
templates=['plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_white', 'plotly_dark',
            'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
data=[]
layout= go.Layout(
        xaxis =  {'showgrid': False,
                    'visible': False,
                    'showticklabels':False,
                    'showline':False,
                    'zeroline': False,
                    'mirror':True,
                    'ticks':None,
                    },
        yaxis =  {'showgrid': False,
                    'visible': False,
                    'showticklabels':False,
                    'showline':False,
                    'zeroline': False,
                    'mirror':True,
                    'ticks':None,
                    },
        newshape={'line_color':None,
                    'fillcolor':None,
                    # 'opacity':0.8,
                    # 'line':{'width':30}
                    },
        template=templates[6],
        font_size=12,
        dragmode='drawopenpath',
        width=580,
        height=630
        )
blank_fig = go.Figure(data, layout)



############ FUNCTIONS
def squash_matrix(df, cols, rows):
    x=0
    col_cut = df.shape[1]//cols
    row_cut = df.shape[0]//rows
    df2 = pd.DataFrame()
    for segment in range(cols):
        df2[segment]=df.iloc[:,x:x+col_cut].mean(axis=1).astype(int)
        x+=col_cut
    df3=df2.groupby(np.arange(len(df))//row_cut).mean().astype(int)
    if len(df3)==rows:
        return df3
    else:
        return df3.iloc[:rows]


def array_to_data_url(img, dtype=None):
    """
    Converts numpy array to data string, using Pillow.
    The returned image string has the right format for the ``image_content``
    property of DashCanvas.
    Parameters
    ==========
    img : numpy array
    Returns
    =======
    image_string: str
    """
    if dtype is not None:
        img = img.astype(dtype)
    df = pd.DataFrame(img)
    df2=squash_matrix(df, cols=28, rows=28) # reduce the number of columns to 28
    return df2


########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle



app.layout = html.Div(children=[
        html.H1('Handwritten Digit Classifier'),
        html.Div([

            html.Div([
                html.H3('Draw & Submit'),
                html.Br(),
                html.Br(),
                html.Br(),
                DashCanvas(
                    id='canvas',
                    lineWidth=10,
                    lineColor='rgba(255, 0, 0, 0.5)',
                    width=canvas_size,
                    height=canvas_size,
                    hide_buttons=["zoom", "pan", "line", "pencil", "rectangle", "undo", "select"],
                    goButtonTitle='Submit',
                ),
            ], style={"padding-left": "20px", "align":"left"}, className="three columns"),

            html.Div([
                html.H3('Image converted to Dataframe', style={"padding-left": "65px", "align":"left"}),
                dcc.Graph(id='output-figure', figure=blank_fig,
                style= {'width': '100%', 'height': '100%', "padding-left": "1px", "align":"left"}
                ),
            ], style={"padding-left": "0px", "align":"left"},
                className='six columns'),

            html.Div([
                html.H3('Predicted Digit'),
                html.Br(),
                html.H4('Random Forest Model:'),
                html.H6(id='rf-prediction', children='...'),
                html.H6(id='rf-probability', children='waiting for inputs'),
                html.Br(),
                html.H4('XGBoost Model:'),
                html.H6(id='xgb-prediction', children='...'),
                html.H6(id='xgb-probability', children='waiting for inputs'),
            ], className='three columns'),
        ], className="twelve columns"),
        html.Br(),
        html.A('Code on Github', href=githublink),
        html.Br(),
        html.A("Data Source", href=sourceurl),
    ], className="twelve columns")


######### CALLBACK
@app.callback(
                Output('output-figure', 'figure'),
                Output('rf-prediction', 'children'),
                Output('rf-probability', 'children'),
                Output('xgb-prediction', 'children'),
                Output('xgb-probability', 'children'),
              Input('canvas', 'json_data'))
def update_data(string):
    if string:
        data = json.loads(string)
        print(data['objects'][0]['path']) # explore the contents of the shape file
        mask = parse_jsonstring(string, shape = (canvas_size, canvas_size))
        img=(255 * mask).astype(np.uint8) # transform the data
        print(img) # explore the transformed data
        array_to_data_output = array_to_data_url(img)
        print(array_to_data_output)

        # display as heatmap
        fig = px.imshow(array_to_data_output, text_auto=True, color_continuous_scale='Blues')
        fig.layout.height = 600
        fig.layout.width = 600
        fig.update(layout_coloraxis_showscale=False)
        fig.update(layout_showlegend=False)

        # convert the user input to the format expected by the model
        some_digit_array = np.reshape(array_to_data_output.values, -1)
        print('some_digit_array',[some_digit_array])
        # make a prediction: Random Forest
        rf_pred = tree_model.predict([some_digit_array])
        rf_prob_array = tree_model.predict_proba([some_digit_array])
        rf_prob = max(rf_prob_array[0])
        rf_prob=round(rf_prob*100,2)

        # make a prediction: XG Boost
        xgb_pred = tree_model.predict([some_digit_array])
        xgb_prob_array = tree_model.predict_proba([some_digit_array])
        xgb_prob = max(xgb_prob_array[0])
        xgb_prob=round(xgb_prob*100,2)

    else:
        raise PreventUpdate


    return   fig,  f'Digit: {rf_pred[0]}', f'Probability: {rf_prob}%', f'Digit: {xgb_pred[0]}', f'Probability: {xgb_prob}%'


if __name__ == '__main__':
    app.run_server(debug=True)
