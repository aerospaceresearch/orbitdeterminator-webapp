# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table_experiments as dt

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
import flask

import math
import base64
from io import StringIO
import sys
import os.path

import orbitdeterminator.kep_determination.ellipse_fit as e_fit
import orbitdeterminator.util.anom_conv as anom_conv
import orbitdeterminator.util.teme_to_ecef as teme_to_ecef
import orbitdeterminator.util.read_data
import orbitdeterminator.util.kep_state
import orbitdeterminator.util.rkf78
import orbitdeterminator.util.golay_window as golay_window
import orbitdeterminator.filters.sav_golay as sav_golay
import orbitdeterminator.filters.triple_moving_average as triple_moving_average
import orbitdeterminator.kep_determination.lamberts_kalman as lamberts_kalman
import orbitdeterminator.kep_determination.interpolation as interpolation
import orbitdeterminator.kep_determination.gibbsMethod as gibbsMethod
import argparse
import matplotlib as mpl
import matplotlib.pylab as plt



app = dash.Dash()
app.title = 'Orbit Determinator'

# setup static folder
STATIC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

@app.server.route('/static/<resource>')
def serve_static(resource):
    return flask.send_from_directory(STATIC_PATH, resource)

app.css.append_css({'external_url': '/static/style.css'})

app.layout = html.Div(children=[

    html.Img(src='/static/logo.png',id='logo'),

    html.H1(children='Orbit Determinator'),

    html.Div('''
        Orbit Determinator: A python package to predict satellite orbits.''',id='subtitle'),
    html.Div(['''
        Choose appropriate units according to .csv file you are going to upload:'''], style={'fontSize': 18,'marginBottom':7},id='unit-selection'),

        html.Div(
          dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Kilometers(km)', 'value': '(km)'},
            {'label': 'Metres(m)', 'value': '(m)'},

        ],
        value='(km)'
    )

     ,style={'width': 250, 'display': 'inline-block'}),




html.Div([''' ''']),
html.Div([''' '''],style={'display': 'inline-block'}),
html.Div([''' ''']),








    html.Div(['''
       Select filter(for pre-processing):'''], style={'fontSize': 22,'marginBottom':7,'display': 'inline-block'},id='filter-selection'),
    html.Div(['''
       Select method for keplerian determination:'''], style={'fontSize': 22,'marginBottom':7,'display': 'inline-block','marginLeft':97},id='filter-selection'),
    html.Div([''' ''']),

    html.Div(
     dcc.Dropdown(
        id='my-dropdown2',
        options=[
            {'label': 'Savintzky Golay', 'value': '(sg)'},
            {'label': 'Triple moving average', 'value': '(tma)'},
            {'label': 'Both', 'value': '(bo)'},
            {'label': 'None', 'value': '(nf)'},

        ],
        value='(sg)'
    )
    ,style={'width': 250, 'display': 'inline-block'}),


    html.Div(
          dcc.Dropdown(
        id='my-dropdown3',
        options=[
            {'label': 'Ellipse Fit', 'value': '(ef)'},
            #{'label': 'Gibbs Method', 'value': '(gm)'},
            {'label': 'Cubic Spline Interpolation', 'value': '(in)'},
            {'label': 'Lamberts Kalman', 'value': '(lk)'}


        ],
        value='(ef)'
    )

     ,style={'width': 350, 'display': 'inline-block','marginLeft': 132}),

    html.Div([''' ''']),
    html.Div([''' '''],style={'display': 'inline-block'}),
    html.Div([''' ''']),
    html.Div([''' '''],style={'display': 'inline-block'}),
    html.Div([''' ''']),

    html.Div(['''
       Input data-- Filter-- Method for keplerian determination-- Orbit of the satellite'''], style={'fontSize': 26,'marginBottom':7,'display': 'inline-block','marginLeft':470},id='filter-selection'),

    html.Div([''' ''']),


    html.Div(id='output-container',style={'fontSize': 0,'marginBottom':15}),


    html.Div(['''Upload a csv file below:'''], style={'fontSize': 22,'marginBottom': 10, 'marginTop': 30}),

    html.Div(id='upload-box',
        children=dcc.Upload(id='file-upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('click to select a file')
            ]),
            multiple=False)),



    html.Div(id='output-div',children=[dt.DataTable(rows=[{}])])



])


@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('my-dropdown', 'value')])


def update_output(value):

    return '- You have selected "{}" as unit of length.'.format(value)





def parse_file(file_content):
    try:
        file_content = file_content.split(',')[1]
        decoded = base64.b64decode(file_content).decode('ascii')
        data = np.loadtxt(StringIO(decoded),skiprows=1)
        assert(data.shape[1] == 4)
        return data
    except Exception as e:
        print(e)
        return np.empty((0,4))

def gen_sphere():
    RADIUS = 6371
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi/2:10j]
    x = (RADIUS*np.cos(u)*np.sin(v)).flatten()
    y = (RADIUS*np.sin(u)*np.sin(v)).flatten()
    z = (RADIUS*np.cos(v)).flatten()
    return x,y,z

def ground_track(kep,time0):
    a = kep[0]
    e = kep[1]
    inc = math.radians(kep[2])
    t0 = math.radians(kep[3])
    lan = math.radians(kep[4])
    tanom = math.radians(kep[5])

    p_x = np.array([math.cos(lan), math.sin(lan), 0])
    p_y = np.array([-math.sin(lan)*math.cos(inc), math.cos(lan)*math.cos(inc), math.sin(inc)])

    # generate 2000 points on the ellipse
    theta = np.linspace(t0+tanom,t0+tanom+4*math.pi,2000)
    radii = a*(1-e**2)/(1+e*np.cos(theta-t0))

    # convert to cartesian
    x_s = np.multiply(radii,np.cos(theta))
    y_s = np.multiply(radii,np.sin(theta))

    # convert to 3D
    mat = np.column_stack((p_x,p_y))
    coords_3D = np.matmul(mat,[x_s,y_s])

    ecc = anom_conv.true_to_ecc(theta,e)
    mean = anom_conv.ecc_to_mean(ecc,e)
    times = anom_conv.mean_to_t(mean,a)
    times += time0

    coords_teme = np.column_stack((times,coords_3D[0],coords_3D[1],coords_3D[2]))
    coords_ecef = teme_to_ecef.conv_to_ecef(coords_teme)

    return coords_ecef,times

@app.callback(Output('output-div','children'),
              [Input('file-upload','contents'),
               Input('file-upload','filename'),
               Input('my-dropdown', 'value'),
               Input('my-dropdown2','value'),
               Input('my-dropdown3','value')])

def display_file(file_content, file_name, dropdown_value,dropdown_value_2,dropdown_value_3):

    if file_content is None:
        return html.Div(['Choose a file to process it.'])

    data_ini = parse_file(file_content)

    if('{}'.format(dropdown_value_2)=='(sg)'):
              window = golay_window.window(10.0, data_ini)
              # Apply the Savintzky - Golay filter with window = 31 and polynomial parameter = 6
              data = sav_golay.golay(data_ini, 31, 6)
              res = data[:, 1:4] - data_ini[:, 1:4]

    elif('{}'.format(dropdown_value_2)=='(tma)'):
              data = triple_moving_average.generate_filtered_data(data_ini, 3)
              res = data[:, 1:4] - data_ini[:, 1:4]

    elif('{}'.format(dropdown_value_2)=='(bo)'):
              data= sav_golay.golay(data_ini, 31, 6)
              data = triple_moving_average.generate_filtered_data(data, 3)
              res = data[:, 1:4] - data_ini[:, 1:4]


    elif('{}'.format(dropdown_value_2)=='(nf)'):
              data=data_ini
              res = data[:, 1:4] - data_ini[:, 1:4]


    #else:
        #data=data_ini






    if (data.shape[0] > 0):
        data_dict = [{'No.':'{:03d}'.format(i+1), 't (s)':data[i][0], 'x '+'{}'.format(dropdown_value):data[i][1], 'y '+'{}'.format(dropdown_value):data[i][2], 'z '+'{}'.format(dropdown_value):data[i][3]} for i in range(data.shape[0])]


        if('{}'.format(dropdown_value_3)=='(ef)'):
            kep, res = e_fit.determine_kep(data[:,1:])


        #elif('{}'.format(dropdown_value_3)=='(gm)'):
            #varchar=1


        elif('{}'.format(dropdown_value_3)=='(in)'):
                 # Apply the interpolation method
                kep_inter = interpolation.main(data)
                # Apply Kalman filters, estimate of measurement variance R = 0.01 ** 2
                kep_final_inter = lamberts_kalman.kalman(kep_inter, 0.01 ** 2)
                kep = np.transpose(kep_final_inter)


        elif('{}'.format(dropdown_value_3)=='(lk)'):
              # Apply Lambert's solution for the filtered data set
                kep_lamb = lamberts_kalman.create_kep(data)
                # Apply Kalman filters, estimate of measurement variance R = 0.01 ** 2
                kep_final_lamb = lamberts_kalman.kalman(kep_lamb, 0.01 ** 2)
                kep = np.transpose(kep_final_lamb)


        # Visuals
        orbit_points = data[:,1:]+res
        earth_points = gen_sphere()

        xyz_org = go.Scatter3d(
            name = 'Original Data',
            legendgroup = 'org',
            showlegend = False,
            x = data[:,1],
            y = data[:,2],
            z = data[:,3],
            mode = 'markers',
            marker = {'size': 1, 'color':'black'}
        )

        xyz_fit = go.Scatter3d(
            name = 'Fitted Ellipse',
            legendgroup = 'fit',
            showlegend = False,
            x = orbit_points[:,0],
            y = orbit_points[:,1],
            z = orbit_points[:,2],
            mode = 'lines',
            line = {'width':5, 'color':'red'}
        )

        xyz_cur = go.Scatter3d(
                name = 'Position at Epoch',
                legendgroup = 'cur',
                showlegend = False,
                x = [orbit_points[0,0]],
                y = [orbit_points[0,1]],
                z = [orbit_points[0,2]],
                mode = 'markers',
                marker = {'size':3, 'color': 'blue'}
        )

        earth_top = go.Mesh3d(
            x = earth_points[0],
            y = earth_points[1],
            z = earth_points[2],
            color = '#1f77b4',
            opacity = 0.5,
            hoverinfo = 'skip',
            showlegend = False
        )

        earth_bottom = go.Mesh3d(
            x = earth_points[0],
            y = earth_points[1],
            z = -earth_points[2],
            color = '#1f77b4',
            opacity = 0.5,
            hoverinfo = 'skip',
            showlegend = False
        )

        xy_org = go.Scatter(
            name = 'Original Data',
            legendgroup = 'org',
            showlegend = True,
            x = data[:,1],
            y = data[:,2],
            mode = 'markers',
            marker = {'size': 5, 'color':'black'}
        )

        xy_fit = go.Scatter(
            name = 'Fitted Ellipse',
            legendgroup = 'fit',
            showlegend = True,
            x = orbit_points[:,0],
            y = orbit_points[:,1],
            mode = 'lines',
            line = {'width':2, 'color':'red'}
        )

        xy_cur = go.Scatter(
            name = 'Position at Epoch',
            legendgroup = 'cur',
            showlegend = True,
            x = [orbit_points[0,0]],
            y = [orbit_points[0,1]],
            mode = 'markers',
            marker = {'size':10, 'color':'blue'}
        )

        yz_org = go.Scatter(
            name = 'Original Data',
            legendgroup = 'org',
            showlegend = False,
            x = data[:,2],
            y = data[:,3],
            mode = 'markers',
            marker = {'size': 5, 'color':'black'}
        )

        yz_fit = go.Scatter(
            name = 'Fitted Ellipse',
            legendgroup = 'fit',
            showlegend = False,
            x = orbit_points[:,1],
            y = orbit_points[:,2],
            mode = 'lines',
            line = {'width':2, 'color':'red'}
        )

        yz_cur = go.Scatter(
                name = 'Position at Epoch',
                legendgroup = 'cur',
                showlegend = False,
                x = [orbit_points[0,1]],
                y = [orbit_points[0,2]],
                mode = 'markers',
                marker = {'size':10, 'color':'blue'}
        )

        xz_org = go.Scatter(
            name = 'Original Data',
            legendgroup = 'org',
            showlegend = False,
            x = data[:,1],
            y = data[:,3],
            mode = 'markers',
            marker = {'size': 5, 'color':'black'}
        )

        xz_fit = go.Scatter(
            name = 'Fitted Ellipse',
            legendgroup = 'fit',
            showlegend = False,
            x = orbit_points[:,0],
            y = orbit_points[:,2],
            mode = 'lines',
            line = {'width':2, 'color':'red'}
        )

        xz_cur = go.Scatter(
                name = 'Position at Epoch',
                legendgroup = 'cur',
                showlegend = False,
                x = [orbit_points[0,0]],
                y = [orbit_points[0,2]],
                mode = 'markers',
                marker = {'size':10, 'color':'blue'}
        )

        xyz_fig = tools.make_subplots(rows=2,cols=2,specs=[[{},{}],
                                                       [{},{'is_3d':True}]])
        xyz_fig.append_trace(xz_org,1,1)
        xyz_fig.append_trace(xz_fit,1,1)
        xyz_fig.append_trace(xz_cur,1,1)
        xyz_fig.append_trace(yz_org,1,2)
        xyz_fig.append_trace(yz_fit,1,2)
        xyz_fig.append_trace(yz_cur,1,2)
        xyz_fig.append_trace(xy_org,2,1)
        xyz_fig.append_trace(xy_fit,2,1)
        xyz_fig.append_trace(xy_cur,2,1)
        xyz_fig.append_trace(xyz_org,2,2)
        xyz_fig.append_trace(xyz_fit,2,2)
        xyz_fig.append_trace(xyz_cur,2,2)
        xyz_fig.append_trace(earth_top,2,2)
        xyz_fig.append_trace(earth_bottom,2,2)

        xyz_fig['layout']['xaxis1'].update(title='x '+'{}'.format(dropdown_value))
        xyz_fig['layout']['yaxis1'].update(title='z '+'{}'.format(dropdown_value), scaleanchor='x')
        xyz_fig['layout']['xaxis2'].update(title='y '+'{}'.format(dropdown_value), scaleanchor='x')
        xyz_fig['layout']['yaxis2'].update(title='z '+'{}'.format(dropdown_value), scaleanchor='x2')
        xyz_fig['layout']['xaxis3'].update(title='x '+'{}'.format(dropdown_value), scaleanchor='x')
        xyz_fig['layout']['yaxis3'].update(title='y '+'{}'.format(dropdown_value), scaleanchor='x3')

        xyz_fig['layout']['scene1']['xaxis'].update(showticklabels=True, showspikes=False, title='x '+'{}'.format(dropdown_value))
        xyz_fig['layout']['scene1']['yaxis'].update(showticklabels=True, showspikes=False, title='y '+'{}'.format(dropdown_value))
        xyz_fig['layout']['scene1']['zaxis'].update(showticklabels=True, showspikes=False, title='z '+'{}'.format(dropdown_value))

        xyz_fig['layout'].update(width=1050, height=700, margin={'t':50})
        xyz_fig['layout']['legend'].update(orientation='h')

        rel_time = data[:,0] - data[0,0]

        xt_org = go.Scatter(
            name = 'Original Data',
            legendgroup = 'org',
            x = rel_time,
            y = data[:,1],
            mode = 'markers',
            marker = {'size': 5, 'color':'black'}
        )

        xt_fit = go.Scatter(
            name = 'Fitted Ellipse',
            legendgroup = 'fit',
            x = rel_time,
            y = orbit_points[:,0],
            mode = 'lines',
            line = {'width':2, 'color':'red'}
        )

        xt_cur = go.Scatter(
            name = 'Position at Epoch',
            legendgroup = 'cur',
            x = [rel_time[0]],
            y = [orbit_points[0,0]],
            mode = 'markers',
            marker = {'size': 10, 'color':'blue'}
        )

        yt_org = go.Scatter(
            name = 'Original Data',
            legendgroup = 'org',
            showlegend = False,
            x = rel_time,
            y = data[:,2],
            mode = 'markers',
            marker = {'size': 5, 'color':'black'}
        )

        yt_fit = go.Scatter(
            name = 'Fitted Ellipse',
            legendgroup = 'fit',
            showlegend = False,
            x = rel_time,
            y = orbit_points[:,1],
            mode = 'lines',
            line = {'width':2, 'color':'red'}
        )

        yt_cur = go.Scatter(
            name = 'Position at Epoch',
            legendgroup = 'cur',
            showlegend = False,
            x = [rel_time[0]],
            y = [orbit_points[0,1]],
            mode = 'markers',
            marker = {'size': 10, 'color':'blue'}
        )

        zt_org = go.Scatter(
            name = 'Original Data',
            legendgroup = 'org',
            showlegend = False,
            x = rel_time,
            y = data[:,3],
            mode = 'markers',
            marker = {'size': 5, 'color':'black'}
        )

        zt_fit = go.Scatter(
            name = 'Fitted Ellipse',
            legendgroup = 'fit',
            showlegend = False,
            x = rel_time,
            y = orbit_points[:,2],
            mode = 'lines',
            line = {'width':2, 'color':'red'}
        )

        zt_cur = go.Scatter(
            name = 'Position at Epoch',
            legendgroup = 'cur',
            showlegend = False,
            x = [rel_time[0]],
            y = [orbit_points[0,2]],
            mode = 'markers',
            marker = {'size': 10, 'color':'blue'}
        )

        rt_org = go.Scatter(
            name = 'Original Data',
            legendgroup = 'org',
            showlegend = False,
            x = rel_time,
            y = (data[:,1]**2+data[:,2]**2+data[:,3]**2)**0.5,
            mode = 'markers',
            marker = {'size': 5, 'color':'black'}
        )

        rt_fit = go.Scatter(
            name = 'Fitted Ellipse',
            legendgroup = 'fit',
            showlegend = False,
            x = rel_time,
            y = ((orbit_points[:,0])**2+
                 (orbit_points[:,1])**2+
                 (orbit_points[:,2])**2)**0.5,
            mode = 'lines',
            line = {'width':2, 'color':'red'}
        )

        rt_cur = go.Scatter(
            name = 'Position at Epoch',
            legendgroup = 'cur',
            showlegend = False,
            x = [rel_time[0]],
            y = [((orbit_points[0,0])**2+
                  (orbit_points[0,1])**2+
                  (orbit_points[0,2])**2)**0.5],
            mode = 'markers',
            marker = {'size': 10, 'color':'blue'}
        )

        t_fig = tools.make_subplots(rows=4,cols=1,shared_xaxes=True)
        t_fig.append_trace(xt_org,1,1)
        t_fig.append_trace(xt_fit,1,1)
        t_fig.append_trace(xt_cur,1,1)
        t_fig.append_trace(yt_org,2,1)
        t_fig.append_trace(yt_fit,2,1)
        t_fig.append_trace(yt_cur,2,1)
        t_fig.append_trace(zt_org,3,1)
        t_fig.append_trace(zt_fit,3,1)
        t_fig.append_trace(zt_cur,3,1)
        t_fig.append_trace(rt_org,4,1)
        t_fig.append_trace(rt_fit,4,1)
        t_fig.append_trace(rt_cur,4,1)

        t_fig['layout']['xaxis1'].update(title='t (s)')
        t_fig['layout']['yaxis1'].update(title='x '+'{}'.format(dropdown_value))
        t_fig['layout']['yaxis2'].update(title='y '+'{}'.format(dropdown_value), scaleanchor='y')
        t_fig['layout']['yaxis3'].update(title='z '+'{}'.format(dropdown_value), scaleanchor='y')
        t_fig['layout']['yaxis4'].update(title='|r| '+'{}'.format(dropdown_value), scaleanchor='y', scaleratio=100)

        t_fig['layout'].update(width=1050, height=700, margin={'t':50})
        t_fig['layout']['legend'].update(orientation='h')

        res_x = go.Histogram(name='Δx', x=res[:,0])
        res_y = go.Histogram(name='Δy', x=res[:,1])
        res_z = go.Histogram(name='Δx', x=res[:,2])

        res_fig = tools.make_subplots(rows=1,cols=3,shared_yaxes=True)
        res_fig.append_trace(res_x,1,1)
        res_fig.append_trace(res_y,1,2)
        res_fig.append_trace(res_z,1,3)

        res_fig['layout']['yaxis1'].update(title='Frequency')
        res_fig['layout']['xaxis1'].update(title='Δx '+'{}'.format(dropdown_value))
        res_fig['layout']['xaxis2'].update(title='Δy '+'{}'.format(dropdown_value))
        res_fig['layout']['xaxis3'].update(title='Δz '+'{}'.format(dropdown_value))

        res_fig['layout'].update(margin={'t':50}, showlegend=False)

        coords_ecef,coord_times = ground_track(kep,data[0][0])

        track = go.Scattergeo(
            name='Ground Trace',
            lat=coords_ecef[:,0],
            lon=coords_ecef[:,1],
            text=coord_times,
            mode='lines',
            line = {'color':'red'})

        track_cur = go.Scattergeo(
            name='Position at Epoch',
            lat=[coords_ecef[0,0]],
            lon=[coords_ecef[0,1]],
            text=coord_times[0],
            marker = {'size':10, 'color':'blue'}
        )

        track_fig = go.Figure(data=[track, track_cur])
        track_fig['layout'].update(height=600, width=1050, margin={'t':50})
        track_fig['layout']['geo']['lataxis'].update(showgrid=True, dtick=30, gridcolor='#ccc')
        track_fig['layout']['geo']['lonaxis'].update(showgrid=True, dtick=60, gridcolor='#ccc')
        track_fig['layout']['legend'].update(orientation='h')

        return [
            dcc.Markdown('''File Name: **'''+file_name+'''**'''),

            html.Details([
                    html.Summary('''File Contents'''),
                    dt.DataTable(rows=data_dict,editable=False)
            ], open=False),

            html.Details([
                html.Summary('''Computed Keplerian Elements'''),
                dt.DataTable(rows=[
                    {'Element':'Semi-major Axis',
                     'Value'            :str(kep[0][0])+'{}'.format(dropdown_value)},
                    {'Element':'Eccentricity',
                     'Value'            :str(kep[1][0])},
                    {'Element':'Inclination',
                     'Value'            :str(kep[2][0])+' °'},
                    {'Element':'Argument of Periapsis',
                     'Value'            :str(kep[3][0])+' °'},
                    {'Element':'Right Ascension of Ascending Node',
                     'Value'            :str(kep[4][0])+' °'},
                    {'Element':'True Anomaly',
                     'Value'            :str(kep[5][0])+' °'},
                    ],editable=False)
            ], open=True),


            html.Details([
                html.Summary('''XYZ Plots'''),
                dcc.Graph(id='xyz-plot', figure=xyz_fig)
            ], open=True),

            html.Details([
                html.Summary('''Ground Track'''),
                dcc.Graph(id='track-plot',figure=track_fig)
            ], open=True),

            html.Details([
                html.Summary('''Time Plots'''),
                dcc.Graph(id='t-plot', figure=t_fig)
            ], open=True),

            html.Details([
                html.Summary('''Residuals'''),
                dcc.Graph(id='res-plot', figure=res_fig),
                dt.DataTable(rows=[
                    {' ':'Maximum','Δx '+'{}'.format(dropdown_value):np.max(res[:,0]),'Δy '+'{}'.format(dropdown_value):np.max(res[:,1]),'Δz '+'{}'.format(dropdown_value):np.max(res[:,2])},
                    {' ':'Minimum','Δx '+'{}'.format(dropdown_value):np.min(res[:,0]),'Δy '+'{}'.format(dropdown_value):np.min(res[:,1]),'Δz '+'{}'.format(dropdown_value):np.min(res[:,2])},
                    {' ':'Average','Δx '+'{}'.format(dropdown_value):np.average(res[:,0]),'Δy '+'{}'.format(dropdown_value):np.average(res[:,1]),'Δz '+'{}'.format(dropdown_value):np.average(res[:,2])},
                    {' ':'Standard Deviation','Δx '+'{}'.format(dropdown_value):np.std(res[:,0]),'Δy '+'{}'.format(dropdown_value):np.std(res[:,1]),'Δz '+'{}'.format(dropdown_value):np.std(res[:,2])}
                ], editable=False)
            ], open=True),
        ]
    else:
        return html.Div(children=[html.Div('''There was an error processing this file.Try uploading another file in required format.''')])

if __name__ == '__main__':
    app.run_server(debug=True)
