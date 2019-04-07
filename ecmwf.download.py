from ecmwfapi import ECMWFDataServer
import numpy as np
server = ECMWFDataServer()
query = {
    "class": "ei",
    "dataset": "interim",
    "date": "2000-01-01/to/2000-12-31",
    "expver": "1",
    "grid": "1.0/1.0",
    "levelist": "300/600/900",
    "levtype": "pt",
    "param": "60.128",
    "step": "0",
    "stream": "oper",
    "time": "00:00:00",
    "type": "an",
    "target": "2000.nc",
    "format": "netcdf",
    "area": "90/0/0/360"
}

def get_pid(items):
    pid = dict()
    pid['pv'] = 60
    pid['gph'] = 129
    pid['temperature'] = 130
    pid['u'] = 131
    pid['v'] = 132
    pid['specific humidity'] = 133
    pid['w'] = 135
    pid['relative vorticity'] = 138
    pid['relative humidity'] = 157
    
    ecm = []
    for item in items:
        ecm.append(pid[item])        
    return ecm

params = ['pv']
levels = [265, 275, 285, 300, 315, 330, 350, 370, 395, 430, 475, 530, 600, 700, 850]
years = np.arange(1979, 2019)

for year in years:
    print year
    query['param'] = get_pid(params)
    query['levelist'] = levels
    query['date'] = str(year)+'-01-01/to/' + str(year) + '-12-31'
    query['target'] = str(year) + '.nc'
    server.retrieve(query)

