import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table_experiments as dt

import numpy as np
import flask

import base64
from io import StringIO
import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir,'orbitdeterminator')))

from kep_determination.ellipse_fit import determine_kep

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
        _data = np.loadtxt(StringIO(decoded),skiprows=1)
        assert(_data.shape[1] == 4)
        return _data
    except Exception as e:
        print(e)
        return np.empty((0,4))

@app.callback(Output('output-div','children'),
              [Input('file-upload','contents'),
               Input('file-upload','filename')])

def display_file(file_content, file_name):

    if file_content is None:
        return html.Div(['Choose a file to display its contents.'])

    data = parse_file(file_content)
    if (data.shape[0] > 0):
        data_dict = [{'No.':'{:03d}'.format(i+1), 't':data[i][0], 'x':data[i][1], 'y':data[i][2], 'z':data[i][3]} for i in range(data.shape[0])]
        kep, res = determine_kep(data[:,1:])
        
        return [html.Div(children=file_name, className='section-titles'),
            
            dt.DataTable(rows=data_dict,editable=False),
            
            html.Div(children='Computed Keplerian Elements', className='section-titles'),

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
                ],editable=False),
            
            dcc.Graph(id='orbit-plot',
                figure={
                    'data':[{'x':data[:,1],'y':data[:,2],'z':data[:,3],'mode':'markers','type':'scatter3d','marker':{
                        'size':2,
                        'opacity':0.8
                        }}],
                    'layout':{
                        'height': 600
                        }
                    })]
    else:
        return html.Div('''There was an error processing this file.''')

if __name__ == '__main__':
    app.run_server(debug=True)
