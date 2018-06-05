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

app = dash.Dash()

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

    html.Div('''Upload a csv file below'''),

    html.Div(id='upload-box',
        children=dcc.Upload(id='file-upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('click to select a file')
            ]),
            multiple=False)),
    
    html.Div(id='output-div',children=[dt.DataTable(rows=[{}])])
])

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

def gen_points(kep):
    a = kep[0]
    e = kep[1]
    inc = math.radians(kep[2])
    t0 = math.radians(kep[3])
    lan = math.radians(kep[4])

    p_x = np.array([math.cos(lan), math.sin(lan), 0])
    p_y = np.array([-math.sin(lan)*math.cos(inc), math.cos(lan)*math.cos(inc), math.sin(inc)])

    # generate 1000 points on the ellipse
    theta = np.linspace(0,2*math.pi,1000)
    radii = a*(1-e**2)/(1+e*np.cos(theta-t0))

    # convert to cartesian
    x_s = np.multiply(radii,np.cos(theta))
    y_s = np.multiply(radii,np.sin(theta))

    # convert to 3D
    mat = np.column_stack((p_x,p_y))
    coords_3D = np.matmul(mat,[x_s,y_s])

    return np.transpose(coords_3D)

def gen_sphere():
    RADIUS = 6371
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi/2:10j]
    x = (RADIUS*np.cos(u)*np.sin(v)).flatten()
    y = (RADIUS*np.sin(u)*np.sin(v)).flatten()
    z = (RADIUS*np.cos(v)).flatten()
    return x,y,z

@app.callback(Output('output-div','children'),
              [Input('file-upload','contents'),
               Input('file-upload','filename')])

def display_file(file_content, file_name):

    if file_content is None:
        return html.Div(['Choose a file to process it.'])

    data = parse_file(file_content)
    if (data.shape[0] > 0):
        data_dict = [{'No.':'{:03d}'.format(i+1), 't':data[i][0], 'x':data[i][1], 'y':data[i][2], 'z':data[i][3]} for i in range(data.shape[0])]
        kep, res = e_fit.determine_kep(data[:,1:])

        # Visuals
        orbit_points = gen_points(kep)
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

        earth_top = go.Mesh3d(
            x = earth_points[0],
            y = earth_points[1],
            z = earth_points[2],
            color = 'blue',
            opacity = 0.5,
            hoverinfo = 'skip',
            showlegend = False
        )

        earth_bottom = go.Mesh3d(
            x = earth_points[0],
            y = earth_points[1],
            z = -earth_points[2],
            color = 'blue',
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

        xyz_fig = tools.make_subplots(rows=2,cols=2,specs=[[{},{}],
                                                       [{},{'is_3d':True}]])
        xyz_fig.append_trace(xz_org,1,1)
        xyz_fig.append_trace(xz_fit,1,1)
        xyz_fig.append_trace(yz_org,1,2)
        xyz_fig.append_trace(yz_fit,1,2)
        xyz_fig.append_trace(xy_org,2,1)
        xyz_fig.append_trace(xy_fit,2,1)
        xyz_fig.append_trace(xyz_org,2,2)
        xyz_fig.append_trace(xyz_fit,2,2)
        xyz_fig.append_trace(earth_top,2,2)
        xyz_fig.append_trace(earth_bottom,2,2)

        xyz_fig['layout']['yaxis1'].update(scaleanchor='x',scaleratio=1)
        xyz_fig['layout']['xaxis2'].update(scaleanchor='x',scaleratio=1)
        xyz_fig['layout']['yaxis2'].update(scaleanchor='x2',scaleratio=1)
        xyz_fig['layout']['xaxis3'].update(scaleanchor='x',scaleratio=1)
        xyz_fig['layout']['yaxis3'].update(scaleanchor='x3',scaleratio=1)
        
        xyz_fig['layout']['xaxis1'].update(title='x (km)')
        xyz_fig['layout']['yaxis1'].update(title='z (km)')
        xyz_fig['layout']['xaxis2'].update(title='y (km)')
        xyz_fig['layout']['yaxis2'].update(title='z (km)')
        xyz_fig['layout']['xaxis3'].update(title='x (km)')
        xyz_fig['layout']['yaxis3'].update(title='y (km)')

        xyz_fig['layout']['scene1']['xaxis'].update(showticklabels=True, showspikes=False, title='x (km)')
        xyz_fig['layout']['scene1']['yaxis'].update(showticklabels=True, showspikes=False, title='y (km)')
        xyz_fig['layout']['scene1']['zaxis'].update(showticklabels=True, showspikes=False, title='z (km)')
        
        xyz_fig['layout'].update(height=700, margin={'t':50})
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
            y = data[:,1]+res[:,0],
            mode = 'lines',
            line = {'width':2, 'color':'red'}
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
            y = data[:,2]+res[:,1],
            mode = 'lines',
            line = {'width':2, 'color':'red'}
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
            y = data[:,3]+res[:,2],
            mode = 'lines',
            line = {'width':2, 'color':'red'}
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
            y = ((data[:,1]+res[:,0])**2+
                 (data[:,2]+res[:,1])**2+
                 (data[:,3]+res[:,2])**2)**0.5,
            mode = 'lines',
            line = {'width':2, 'color':'red'}
        )

        t_fig = tools.make_subplots(rows=4,cols=1,shared_xaxes=True)
        t_fig.append_trace(xt_org,1,1)
        t_fig.append_trace(xt_fit,1,1)
        t_fig.append_trace(yt_org,2,1)
        t_fig.append_trace(yt_fit,2,1)
        t_fig.append_trace(zt_org,3,1)
        t_fig.append_trace(zt_fit,3,1)
        t_fig.append_trace(rt_org,4,1)
        t_fig.append_trace(rt_fit,4,1)

        t_fig['layout']['xaxis1'].update(title='t (s)')
        t_fig['layout']['yaxis1'].update(title='x (km)')
        t_fig['layout']['xaxis2'].update(title='t (s)')
        t_fig['layout']['yaxis2'].update(title='y (km)')
        t_fig['layout']['xaxis3'].update(title='t (s)')
        t_fig['layout']['yaxis3'].update(title='z (km)')
        t_fig['layout']['xaxis4'].update(title='t (s)')
        t_fig['layout']['yaxis4'].update(title='|r| (km)')

        t_fig['layout'].update(height=700, margin={'t':50})
        t_fig['layout']['legend'].update(orientation='h')

        return [
            dcc.Markdown('''File Name: **'''+file_name+'''**'''),

            html.Details([
                    html.Summary('''File Contents'''),
                    dt.DataTable(rows=data_dict,editable=False)
            ]),
            
            html.Details([
                html.Summary('''Computed Keplerian Elements'''),
                dt.DataTable(rows=[
                    {'Element':'Semi-major Axis',
                     'Value'            :str(kep[0][0])},
                    {'Element':'Eccentricity',
                     'Value'            :str(kep[1][0])},
                    {'Element':'Inclination',
                     'Value'            :str(kep[2][0])},
                    {'Element':'Argument of Periapsis',
                     'Value'            :str(kep[3][0])},
                    {'Element':'Right Ascension of Ascending Node',
                     'Value'            :str(kep[4][0])},
                    {'Element':'True Anomaly',
                     'Value'            :str(kep[5][0])},
                    ],editable=False)
            ],open=True),
            
            html.Details([
                html.Summary('''XYZ Plots'''),
                dcc.Graph(id='xyz-plot', figure=xyz_fig)],
            open=False),

            html.Details([
                html.Summary('''Time Plots'''),
                dcc.Graph(id='t-plot', figure=t_fig)],
            open=True)
        ]
    else:
        return html.Div('''There was an error processing this file.''')

if __name__ == '__main__':
    app.run_server(debug=True)
