# %%
# !pip install pandas
# !pip install plotly
# !pip install dash_core_components dash_html_components
# !pip install plotly.express
# !pip install -U scikit-image
# !pip install --upgrade jupyter_http_over_ws>=0.0.7 && jupyter serverextension enable --py jupyter_http_over_ws
# !pip install dash_canvas
# !pip install ipynb-py-convert
# !pip install image-utils
# !pip install nilearn
# !pip install shapes
# !pip install pybase64
# !pip install shapes
# !pip install boto3
# !pip install python-dotenv
# !pip install dash_bootstrap_components
# !pip install pillow==7.0.0
#!pip install tensorflow

# %%
# pip list

# %%
# https://github.com/plotly/dash-3d-image-partitioning/blob/master/app.py
# https://dash-gallery.plotly.host/dash-3d-image-partitioning/

# %%
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.express as px
from nilearn import image
import nibabel as nib
from sys import exit
import image_utils
import numpy as np
import skimage
import base64
import dash
import time
import os
import io
import dash_bootstrap_components as dbc
from dash_canvas.utils import array_to_data_url
from skimage import data, img_as_ubyte, segmentation, measure
from dash.dependencies import Input, Output, State, ClientsideFunction

# %%
from dash.dependencies import Input, Output, State
from dotenv import load_dotenv, find_dotenv
import json
import time
import uuid
from copy import deepcopy
import boto3
import requests
#from flask_caching import Cache

# import dash_reusable_components as drc
# from utils import STORAGE_PLACEHOLDER, GRAPH_PLACEHOLDER, IMAGE_STRING_PLACEHOLDER
# from utils import apply_filters, show_histogram, generate_lasso_mask, apply_enhancements

# %%
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go 
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage
import nibabel as nib
import numpy as np
import operator

# %%
# os.getcwd()  # TODO: REMOVE ME

# %%
### self-made frameworks
import K8L_detect

# %%
from flask import Flask, Request
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.getcwd() + '/MRI'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app = Flask(__name__)  # <---------  WTF??
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# %%
def alert_():
    alert = dbc.Alert(
        [
            html.H4("Well done!", className="alert-heading"),
            html.P(
                "This is a success alert with loads of extra text in it. So much "
                "that you can see how spacing within an alert works with this "
                "kind of content."
            ),
            html.Hr(),
            html.P(
                "Let's put some more text down here, but remove the bottom margin",
                className="mb-0",
            ),
        ]
    )
    return alert

# %%
def plot_3d(image, threshold=-300):
    fig = go.Figure()
    p = image
    # p = image.transpose(2,1,0)
    p = ndimage.rotate(p, 180, axes=(1,0))
    p = ndimage.zoom(p, (0.5, 0.5, 1), order= 3)
    # p = image
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig_ = plt.figure(figsize=(10, 10))
    ax = fig_.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
    return fig
# url = "/run/user/1000/gvfs/google-drive:host=gmail.com,user=svetloffsergey/1XG17oqVBvn4Lusj4Yns-cFzWYx5DBWMT/1fcAJfhwZvXw2emzciyVp8SoN9EqxYXDA/1abY2afPM745jnKNGmKXA4BZgb6acjXQt"
# # image = "/content/drive/MyDrive/JN/NN/COVID_detection_picture/sick_7c7160149aec1ebf15b28166f5458c49.nii"
# image = "/sick_7c7160149aec1ebf15b28166f5458c49.nii"
# image = nib.load(url + image);
# image = image.get_data();
# image.shape
# fig = go.Figure(plot_3d(image, threshold=-300))
# fig.show()

# %%
# layout = dict(
#     autosize=True,
#     automargin=True,
#     margin=dict(
#         l=30,
#         r=30,
#         b=20,
#         t=40
#     ),
#     hovermode="closest",
#     plot_bgcolor="#F9F9F9",
#     paper_bgcolor="#F9F9F9",
#     legend=dict(font=dict(size=10), orientation='h'),
#     title='Satellite Overview',
# #     mapbox=dict(
# #         accesstoken=mapbox_access_token,
# #         style="light",
# #         center=dict(
# #             lon=-78.05,
# #             lat=42.54
# #         ),
# #         zoom=7,
# #     )
# )

# %%
# app = dash.Dash()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# %%

def plot_image(path):
    fig = go.Figure();
#     plt.figure(figsize=(12, 12));
    # path = image_fetcher.fetch('data/cells.tif')
    # path = image_fetcher.fetch('data/cells.tif')
    try:
        if path != None:
            path = path
    except:
        print(f"Default image:")
        path = '/С:/Users/Admin/Documents/JN/NN/COVID_detection_picture/MRI/sick_7c7160149aec1ebf15b28166f5458c49.nii'
#     print(type(path))
    data = nib.load(path);
    print(type(data))
    
    data = data.get_data();
    print(type(data[0]), data[0].shape)
    data = data.transpose(2,1,0)
    print(type(data), data.shape)
    
    data = ndimage.zoom(data, (1, 1, 1), order = 1)
    print(data.shape)
    img = data[:]
    fig = px.imshow(img, 
                    animation_frame = 0, 
                    binary_string = True, 
                    labels = dict(animation_frame = "slice"),    
                    width = 512,
                    height = 512, 
#                     title = "NIFTI Detection scanner"
                   )
#     fig.show()    
    return fig
# plot_image(path)

# %%
BANNER = html.Div(
    [html.H3('--Neural net detection--'),
     html.H3(' <COVID 19> from MRI-files type--',
             id='title'),
     html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"
         ,  style={'height':'11%', 'width':'11%'})],
    className="row", ### oldName 'banner' - ok!
    style={'width': '100%', 
           'height' : '50%',
           'lineHeight': '50px', 
           'borderWidth': '1px', 
#            'borderStyle': 'dashed', 
           'borderRadius': '5px',
           'textAlign': 'center',
          'float' : 'center',
          'textAlign' : 'center'}
, ) 

# %%
CLASS_NAME_HEADER = 'two columns'

# %%
HEADER_LEFT = html.Div(
    className = 'four columns',
    children=[
        html.Div(className = 'pretty_container',
                 children = [
        dcc.Upload(id='upload_image', 
                   children=['Drag and Drop or .. ', 
                             html.A('Select an Image')]),
                   ],
#                    style={'width': '50%', 
#                           'height': '50px', 
#                           'lineHeight': '50px', 
#                           'borderWidth': '1px', 
# #                           'borderStyle': 'dashed', 
#                           'borderRadius': '5px',
#                           'textAlign': 'center'},
                  )
    ])

# %%
HEADER_CENTER  = html.Div(
    className = 'four columns', 
    children = [
        html.Div(className = 'pretty_container',
                 children = [
        dcc.RadioItems(
            id = 'select_model',
            options = [
#                 {'label': ' ResNet50', 'value': 'ResNet50'},
                {'label': ' Keras', 'value': 'Keras'}
                      ],
            value ='Keras', 
            labelStyle = {'margin-right': '5px',
                         'display': 'inline-block'}
        ),
        html.Div([
            html.Button('Run Detection',
                        id='button_run_operation',
                        style = {'margin-right': '10px', 'margin-top': '5px'},
                        n_clicks = 0
                       ),
            html.Button('Undo', id='button_undo',
                        style={'margin-top': '5px'},
#                        n_clicks = 0
                       ),
        ])    
    ]),
                 ],
#     style={'width': '50%', 
#            'height': '50px', 
#            'lineHeight': '50px',
#            'borderWidth': '1px', 
# #            'borderStyle': 'dashed', 
#            'borderRadius': '5px',
#            'textAlign': 'center'}
)

# %%
HEADER_CLASS = html.Div(
    className = 'two columns', #old four -ok!
    children = [
        html.Div([
        html.Div([
            html.P('Class:'),
            html.H6(id="well_text", className="info_text"),]),
        ],
            className = 'pretty_container'
        )],
    id="wells_class",
#     style={'width': '100%', 
#            'height' : '50%',
#            'lineHeight': '50px', 
#            'borderWidth': '1px', 
# #            'borderStyle': 'dashed', 
#            'borderRadius': '5px',
#            'textAlign': 'left'}
)

# %%
HEADER_RATE = html.Div(
    className = 'two columns', #old four -ok!
    children = [
        html.Div([
        html.Div([
            html.P('Rate:'), 
            html.H6(id="well_rate", className="info_text"),]),
        ],
            className = 'pretty_container'
        )],
    id="wells_rate",
#     style={'width': '100%', 
#            'height' : '50%',
#            'lineHeight': '50px', 
#            'borderWidth': '1px', 
# #            'borderStyle': 'dashed', 
#            'borderRadius': '5px',
#            'textAlign': 'left'}
)

# %%
MAIN_SCREEN = html.Div(className = 'row', 
                       children = [
                        html.Div(
                            className='pretty_container',
                            children=[
                            html.H6('File name:', id = 'well_file'),
                            dcc.Graph(id = 'id_display_view',
                                     )
                            ],
                        )
                       ],
                       style={
                           'width': '80%',
                           'display': 'inline-block', 
                           'padding': '20 20',
                           'float': 'center'
                       }
                      )

# %%
HEADER = html.Div(
    className = 'row',
    children = [
        html.Div(
            HEADER_LEFT),
        html.Div(
            HEADER_CENTER),
        html.Div(
            HEADER_CLASS),
        html.Div(
            HEADER_RATE
                )
    ],
    style={
        'width': '80%',
        'display': 'inline-block',
#         'padding': '20 20',
#         'float': 'center'
    }
)

# %%
app.layout = html.Div([
    dcc.Store(id='aggregate_data', ),
    BANNER,
    HEADER,
    MAIN_SCREEN
],
    style={
        "display": "flex",
        "flex-direction": "column"
    })

# %%
path_to_image = "sick_7c7160149aec1ebf15b28166f5458c49.nii"
# os.path.abspath('MRI/' + path_to_image)  # TODO: REMOVE ME?

# %%
def fetch_aggregate_(full_path, model):
    click = 0
    try:
        full_path == None
    except:
#         full_path = os.getcwd() + "MRI/test_MRI.nii"
        full_path = os.getcwd() + "/MRI/sick_7ff18f5d3de11b9ae7a9e5d651313fbd.nii"
#     fig = plot_image(full_path)
    if model == 'Keras':
#         full_path = os.getcwd() + full_path
        predict = K8L_detect.main(full_path)
    elif model == 'ResNet50':
        pass
#         predict = RN50_detect.main(full_path)
#         predict = max(predict.items(), key = operator.itemgetter(1))
        
    return predict

### get fetch ###
@app.callback(
#     Output('aggregate_data', 'data'),
    Output('aggregate_data', 'data'),
    [Input('upload_image', 'filename'), Input('select_model', 'value'), Input('button_run_operation', 'n_clicks')])
def get_fetch_(filename, select_model, click):
#     dir_path = os.getcwd()
    try:
        filename is None
    except:
        return print('Error path', type(filename))
    
    full_path = os.path.abspath('MRI/' + filename)
    if filename != None and select_model != None and click != 0:
        predict = fetch_aggregate_(full_path, select_model)
        click = 0
        return predict



### get slicer ###
@app.callback(
#     Output('aggregate_data', 'data'),
    Output('id_display_view', 'figure'),
    [Input('upload_image', 'filename'), Input('select_model', 'value'), Input('button_run_operation', 'n_clicks')])
def get_image_(filename, select_model, click):
#     dir_path = os.getcwd()
    try:
        filename is None
    except:
        return print('Error path', type(filename))
    full_path = os.path.abspath('MRI/' + filename)
    if filename != None:
        fig = plot_image(full_path)
        return fig
    
### get responce from def fetch_aggregate ###
@app.callback(
    Output('well_text', 'children'),
    Input('aggregate_data', 'data'))
def predict_data_class(data):
        return data[0]

### get responce from def fetch_aggregate ###
@app.callback(
    Output('well_rate', 'children'),
    Input('aggregate_data', 'data'))
def predict_data_rate(data):
        return str(round(float(data[1]) * 100, 2)) + "%"

### get filename
@app.callback(
    Output('well_file', 'children'),
    [Input('upload_image', 'filename'), Input('select_model', 'value'), Input('button_run_operation', 'n_clicks')])
def get_name_file_(name, model, click):
    if name != None:
#             return ["File: " + os.fspath(name), " | Model: " + model, click]
            return ["File: " + os.path.abspath(name), " | Model: " + model, click]
#             return ["File: " + os.getcwd() + name, " | Model: " + model, click]
    else:
        alert()

# %%
if __name__ == "__main__":
    !ipynb-py-convert app.ipynb app.py
    app.run_server(debug=False);
