# LXeSlowControls

This repository contains Python code to parse log files written by the LabVIEW Slow Controls VIs as well as some basic plotting tools.

## Dependencies

A few basic Python packages are required:

* numpy
* matplotlib
* csv
* pickle
* pandas

## Usage

Start by creating a `SlowControls` object with a list of log files and column maps

```python
import yaml
from SlowControls import SlowControls

datasets = ['sample_logfile.dat'] # list of datasets to be parsed

with open('colmap.yml','r') as infile:
    colmap = yaml.safe_load(infile) # column map stored as a yaml file

# create SlowControls object and load data from log file based on column maps
SC = SlowControls(datasets,indices=colmap['indices'],labels=colmap['labels'])

# do some plotting...
```

An example notebook displaying more functionality will be added in the future.
