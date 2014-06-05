import json
import numpy as np
from mayavi.mlab import *
import pylab as pl
import sqlite3
import os

norm = np.linalg.norm
sin = np.sin
cos = np.cos

DBNAME = '3dinteraction.sqlite3'

from UserDict import DictMixin
class Scene(DictMixin):
    def __init__ (self, record):
        fields = ('id', 'subject', 'input', 'output', 'startTime', 'endTime',
                  'duration', 'numInputs', 'startPos', 'endPos', 'inputs')
        scene = dict(zip(fields, record))
        scene['startPos'] = np.frombuffer(scene['startPos'], dtype=np.float64)
        scene['endPos'] = np.frombuffer(scene['endPos'], dtype=np.float64)
        scene['inputs'] = np.frombuffer(scene['inputs'], dtype=np.float32).reshape(scene['numInputs'], 9)
        self.__dict__.update(scene)

    def __getitem__ (self, key):
        return self.__dict__[key]

    def __setitem__ (self, key, value):
        self.__dict__[key] = value

    def __delitem__ (self, key):
        return self.__dict__.pop(key)

def RotM_to_AA (R):
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/
    theta = arccos((R.trace()-1) / 2)                          # rotation angle
    d1 = R[2,1] - R[1,2]
    d2 = R[0,2] - R[2,0]
    d3 = R[1,0] - R[0,1]
    dist = sqrt(d1**2 + d2**2 + d3**2)
    axis = 1/dist * np.array([[d1], # rotation as axis angle
                              [d2],
                              [d3]])
    return theta * axis
    
def RotAA_to_M (AA):
    # http://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Conversion_to_rotation_matrix
    if AA.shape != (3,1):
        AA = AA.reshape(3,1)
    theta = norm(AA)                     # angle of rotation
    k = AA/theta                         # unit length axis of rotation
    kx = np.array([[   0, -k[2],  k[1]], # skew symmetric matrix for cross product
                   [k[2],     0, -k[0]],
                   [-k[1], k[0],  0   ]])
    R = cos(theta)*np.eye(3) + sin(theta)*kx + (1-cos(theta))*k.dot(k.T)
    return R

def createDB():
    db = sqlite3.connect(DBNAME)
    db.execute('DROP TABLE IF EXISTS Experiments')
    db.execute('''
CREATE TABLE Experiments
(id integer primary key asc autoincrement not null,
 subject text,
 input text,
 output text,
 startTime real,
 endTime real,
 duration real,
 numInputs integer,
 startPos blob,
 endPos blob,
 inputs blob)
''')
    db.commit()
    db.close()
    
def openDB():
    return sqlite3.connect(DBNAME)

def addToDB(metadata):
    db = sqlite3.connect(DBNAME)
    c = db.cursor()
    rows = []
    config = metadata['config']
    for scene in metadata['scenes']:
        rows.append((config['subject'],
                     config['input'],
                     config['output'],
                     scene['startTime'],
                     scene['endTime'],
                     scene['duration'],
                     scene['numInputs'], 
                     np.getbuffer(scene['startPos']),
                     np.getbuffer(scene['endPos']),
                     np.getbuffer(scene['inputs'])
                     ))
    c.executemany('insert into experiments (subject, input, output, startTime, endTime, duration, numInputs, startPos, endPos, inputs) values (?,?,?,?,?,?,?,?,?,?)', rows)
    db.commit()
    db.close()
    return
    
def parseTrial(filename):
    '''
    Loads and normalizes data for a single trial, e.g. name_TIMESTAMP.json
    '''
    # Convert windows style directory separators to unix style
    raw = open(filename).read().replace('\\', '/')
    metadata = json.loads(raw)

    for s in metadata['scenes']:
        # Format for the binary data is 9 columns:
        # timestamp, x,y,z, rx,ry,rz,rw, flags
        rows = s['numInputs']
        inputs = np.fromfile(s['inputs'][4:],dtype=np.float32)
        inputs = inputs.reshape(rows, 9)
        s['duration'] = float(inputs[-1,0] - inputs[0,0])

        sp = s['startPos']
        ep = s['endPos']
        p1 = np.array([sp['px'], sp['py'], sp['pz']]) # start pos
        p2 = np.array([ep['px'], ep['py'], ep['pz']]) # end pos
        
        s['startPos'] = p1      # store originals
        s['endPos'] = p2

        T = p1.copy()           # translation
        p1 -= T                 # set as p1 as origin
        p2 -= T
        l = norm(p2)

        # Figure out rotation to get p2 to lie along the x axis
        # axis = i X p2
        axis = np.array([0, p2[2], -p2[1]])
        axis /= norm(axis)      # normalize

        # angle of rotation is acos(<p2, i> / (|p2|*|i|))
        angle = np.arccos(p2[0] / l)

        rotM = RotAA_to_M(angle*axis)
        p2 = rotM.dot(p2.reshape(3,1)).reshape(3) # reshaping for matrix math

        # Transform input positions
        for row in xrange(rows):
            pos = inputs[row, 1:4]
            inputs[row, 1:4] = rotM.dot((pos - T).reshape(3,1)).reshape(3)
        s['inputs'] = inputs
    return metadata

def populateDB(directory = '.', reset=True):
    files = [f for f in os.listdir(directory) if '.json' in f]
    if reset:
        createDB()
    count = 0
    for f in sorted(files):
        print 'Parsing', f
        metadata = parseTrial(directory + "/" + f)
        addToDB(metadata)
    return

def sceneFromRecord (record):
    return Scene(record)

def getScenes (criteria = None):
    db = openDB()
    sql = "Select * from Experiments"
    if criteria is not None:
        sql += " where " + criteria
    scenes = map(sceneFromRecord, db.execute(sql).fetchall())
    db.close()
    return scenes

def plotScene (scene, useColor=False, *args, **kwargs):
    i = scene['inputs']
    pos = i[:,1:4]

    # VTK won't render coincident points, so we have to get rid of overlaps
    points = [pos[0]]
    for idx in xrange(1,pos.shape[0]):
        v = pos[idx] - pos[idx-1]
        if v.dot(v) > 0.0001:
            points.append(pos[idx])
    points = np.array(points)
    
    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    if useColor:
        S = [1,0.5] * (len(X) / 2 + 1)
        if len(S) > len(X):
            S = S[:len(X)]
        plot3d(X,Y,Z,S, *args, **kwargs)
    else:
        plot3d(X,Y,Z, *args, **kwargs)
    points3d(X,Y,Z, scale_factor=0.1, resolution=4)

def cylinderCompress(scene):
    '''
    rotates all points to lie on the XY plane
    '''
    P = scene['inputs'][:,1:4]
    i_axis = np.array([1,0,0])
    for idx in xrange(P.shape[0]):
        p = P[idx]
        theta = np.arccos(p[1] / np.sqrt(p[1]**2 + p[2]**2))
        if p[2] != 0:
            sign = p[2] / np.abs(p[2])
        rotM = RotAA_to_M(-1 * sign * theta * i_axis)
        P[idx] = rotM.dot(p)
    
