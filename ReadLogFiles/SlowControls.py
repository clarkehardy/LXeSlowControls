import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import matplotlib.dates as mdates
import pandas as pd
import time
import pickle
from tqdm import tqdm, trange


class SlowControls(object):
    def __init__(self, datasets = None, system = None, labels = None, indices = None, logfiles = None):
        """
        Imports a set of slow controls data files
        produced by LabVIEW and initializes a pandas
        dataframe with labelled columns.

        The labels and indices are tied intimately to the output 
        data file of the labview program. this also allows for the
        use of this same class for multiple different systems (LS, SS, purifier).
        If no labels or indices are passed, it will use some known values from the LS. 
        """
        start = time.time()
        if datasets==None:
            print('\nCreating an empty SlowControls object.')
            return
        
        if labels==None:
            first_name = datasets[0].split('/')[-1]
            system = first_name[0:-20]
            if system=='SS':
                self._LabVIEW_labels = ['date','T_in','T_out','T_Cu_top','T_Cu_bottom','T_cell_mid','P_XP3','P_XP5','P_CCG','cool_on']
                self._LabVIEW_indices = np.array([0,8,7,5,1,3,22,23,24,35])
            elif system=='LS':
                self._LabVIEW_labels = ['date','T_Cu_bottom','T_cell_top','T_cell_mid','T_cell_bottom','T_Cu_top','T_ambient','T_in','T_top_flange','T_set_min','T_set_max','mass_flow','P_XP3','P_XP5','P_CCG','cool_on','water_flow','discharge_pressure','suction_pressure','coil_in_temp','coil_out_temp','water_temp','fridge_CT','coil2_out_temp']
                self._LabVIEW_indices = np.array([0,1,2,3,4,5,6,7,8,10,11,21,22,23,24,35,38,39,40,41,42,43,44,45])
            elif system=='Purifier':
                self._LabVIEW_labels = ['date','set_temp','heater_on','T_Al_middle','T_Al_top','T_Al_bottom','T_upper','T_top_flange','T_lower','T_bottom_flange']
                self._LabVIEW_indices = np.array([0,1,2,3,4,5,6,7,8,9])

        else:
            self._LabVIEW_labels = labels
            self._LabVIEW_indices = indices

        self.system = system #label of the system, like "SS", "LS", "Purifier" (and typically file prefix of datafiles)
        datasets, = self._FormatArgs((datasets,))
        self._num_datasets = len(datasets)
        points = []
        cols = 0
        abort = False
        for i in range(self._num_datasets):
            if abort:
                break
            print('\nOpening dataset in {}...'.format(datasets[i]))
            with open(datasets[i],'r') as infile:
                reader = csv.reader(infile,delimiter='\t')
                looper = tqdm(reader, desc="Processing lines...")
                for line in looper:
                    old_cols = cols
                    cols = len(line)
                    if (i>0) & (cols!=old_cols):
                        print('Warning: file '+datasets[i]+' has {} columns, but'.format(cols))
                        print('previous file '+datasets[i-1]+' had {} columns.'.format(old_cols))
                        print('These should be loaded separately with separate column maps. Aborting...')
                        abort = True
                        break
                    points.append(pd.DataFrame([np.array(line)[self._LabVIEW_indices].astype(float)],
                                               columns=self._LabVIEW_labels,index=[str(i+1)]))
        
        # conversion between NI timestamp and unix timestamp
        ni_time_offset = 2082844800

        # load notes from the log files
        if logfiles==None:
            logfiles = []
        log_dates = []
        log_notes = []
        for log in logfiles:
            with open(log,'r') as infile:
                reader = csv.reader(infile,delimiter='\t')
                for line in reader:
                    log_dates.append(float(line[0]) - ni_time_offset)
                    log_notes.append(line[1])

        for point in points:
            point['notes'] = ''

        self._data = pd.concat(points,ignore_index=True)
        self._data['date'] = self._data['date'] - ni_time_offset
        if('P_CCG' in self._data):
            self._data['P_CCG'] = self._data['P_CCG']*1e-6 #i don't like this, can we change somehow?

        # associate the notes with the correct point in the data file
        date_matches = [np.argmin(np.abs(self._data['date'] - log_date)) for log_date in log_dates]
        for match in date_matches:
            self._data.loc[match,'notes'] = log_notes[date_matches.index(match)]

        min_date = min(self._data['date'])
        max_date = max(self._data['date'])
        print('\nFound {} readings between {} and {}.'.format(len(self._data),\
                                                              datetime.fromtimestamp(min_date).strftime('%m/%d/%Y %H:%M:%S'),\
                                                              datetime.fromtimestamp(max_date).strftime('%m/%d/%Y %H:%M:%S')))

        print("\nSorting by date...")
        self._data = self._data.sort_values("date")
        print('\nCreated pandas dataframe containing slow controls data.')
        print('\nTime: {:.3f} seconds.'.format(time.time()-start))
        
    def _GetDateInterval(self,start_date,end_date):
        """
        Returns a slice that can be used to index for
        a given range of dates.
        """
        start_date,end_date = self._FormatArgs((start_date,end_date))
        ranges = []
        for i in range(len(start_date)):
            start = datetime.timestamp(start_date[i])
            end = datetime.timestamp(end_date[i])
            dates = self._data['date']
            try:
                start_index = [date>start for date in dates].index(True)
            except ValueError:
                start_index = 0
            try:
                end_index = [date>end for date in dates].index(True)
            except ValueError:
                end_index = -1
            ranges.append(slice(start_index,end_index,1))
        return ranges
            
    def _GetHoursFromTime(self,start_time,end_time):
        """
        Returns a list containing arrays of the number
        of hours since a specified starting time.
        """
        start_time,end_time = self._FormatArgs((start_time,end_time))
        ranges = self._GetDateInterval(start_time,end_time)
        hours_array = []
        for i in range(len(start_time)):
            secs_from_start = np.array(self._data['date'][ranges[i]])-self._data['date'].iloc[ranges[i].start]
            hours_array.append(secs_from_start/3600.)
        return hours_array,ranges

    def GetLabels(self):
        """
        Returns labels and indices
        """
        return self._LabVIEW_labels, self._LabVIEW_indices

    def PlotVsDate(self,quantity,start_date,end_date,labels=[],title='',ylabel='',semilogy=False):
        """
        Method for plotting a given quantity or set of quantities
        over a specified date range.
        """
        if(quantity == []):
            fig, ax = plt.subplots()
            return fig, ax 
            
        quantity, = self._FormatArgs((quantity,))
        if labels==[] and type(start_date)==list:
            labels = list(range(1,len(start_date)+1))
            labels = list(map(str,labels))
        elif labels==[] and type(start_date)!=list:
            labels = ['']
        start_date,end_date,labels = self._FormatArgs((start_date,end_date,labels))
        ranges = self._GetDateInterval(start_date,end_date)
        if ylabel=='':
            ylabel = quantity[0]
        fig,ax = plt.subplots()
        for i in range(len(start_date)):
            dates = self._data['date'][ranges[i]]
            dates = [datetime.fromtimestamp(date) for date in dates]
            for j in range(len(quantity)):
                yvals = self._data[quantity[j]][ranges[i]]
                if semilogy:
                    ax.semilogy(dates,yvals,label=quantity[j]+' '+labels[i])
                else:
                    ax.plot(dates,yvals,label=quantity[j]+' '+labels[i])
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y %H:%M'))
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        fig.tight_layout()
        return fig,ax

    def PlotVsTime(self,quantity,start_date,end_date,labels=[],title='',ylabel='',semilogy=False):
        """
        Method for plotting a given quantity or set of quantities
        over time since a specified start time. Useful for comparing
        behavior directly among different circumstances.
        """
        quantity, = self._FormatArgs((quantity,))
        if labels==[] and type(start_date)==list:
            labels = list(range(1,len(start_date)+1))
            labels = list(map(str,labels))
        elif labels==[] and type(start_date)!=list:
            labels = ['']
        start_date,end_date,labels = self._FormatArgs((start_date,end_date,labels))
        hours,ranges = self._GetHoursFromTime(start_date,end_date)
        if ylabel=='':
            ylabel = quantity[0]
        fig,ax = plt.subplots()
        for i in range(len(start_date)):
            for j in range(len(quantity)):
                yvals = self._data[quantity[j]][ranges[i]]
                if semilogy:
                    ax.semilogy(hours[i],yvals,label=quantity[j]+' '+labels[i])
                else:
                    ax.plot(hours[i],yvals,label=quantity[j]+' '+labels[i])
        ax.set_title(title)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        fig.tight_layout()
        return fig,ax

    def GetDateArray(self,start_date,end_date):
        """
        Returns an array of the dates between a given start
        date and end date.
        """
        ranges = self._GetDateInterval(start_date,end_date)
        dates = self._data['date'][ranges[0]]
        dates = [datetime.fromtimestamp(date) for date in dates]
        return np.array(dates)

    def GetHoursArray(self,start_date,end_date):
        """
        Returns an array of the hours since the start time between
        a given start time and end time.
        """
        hours,ranges = self._GetHoursFromTime(start_date,end_date)
        return hours[0]

    def GetQuantityArray(self,quantity,start_date,end_date):
        """
        Returns an array of a specified quantity between a given
        start date and end date.
        """
        ranges = self._GetDateInterval(start_date,end_date)
        return np.array(self._data[quantity][ranges[0]])
    
    def GetNotableDates(self):
        """
        Returns a list of dates that have notes associated with them.
        """
        dates = [datetime.fromtimestamp(d) for d in self._data[self._data['notes'] != '']['date']]
        return dates
    
    def _FormatArgs(self,args):
        """
        Converts arguments to a list. Allows for either a list
        or a single object to be passed as arguments for most
        private methods.
        """
        new_args = []
        for arg in args:
            if type(arg) is not list:
                new_args.append([arg])
            else:
                new_args.append(arg)
        return tuple(new_args)

    def PrintDataFormat(self):
        """
        Prints the format of the slow controls data file produced by LabVIEW.
        """
        for i in range(len(self._LabVIEW_labels)):
            print('index: {}, quantity: {}'.format(self._LabVIEW_indices[i],str(self._LabVIEW_labels[i])))
        print()
    
    def PrintHead(self):
        """
        Prints the pandas DataFrame head.
        """
        print(self._data.head())

    def PrintTimeBounds(self):
        """
        Prints the start and end date for the data.
        """
        dates = self._data['date']
        print("From ",end='')
        print(datetime.fromtimestamp(dates.min()), end='')
        print(" to ", end='')
        print(datetime.fromtimestamp(dates.max()))

    def GetTimeBounds(self):
        """
        Returns the start and end date for the data.
        """
        dates = self._data['date']
        return datetime.fromtimestamp(dates.min()), datetime.fromtimestamp(dates.max())

    def GetMostRecentValues(self):
        """
        Gets the most recent values for each process variable (quantity).
        """
        label_val_tuples = [] #[[label, val], [label, val], ...]
        for l in self._LabVIEW_labels:
            label_val_tuples.append([l, self._data[l][-1]])
        return label_val_tuples

    def SaveDataframe(self,filename='data.pkl'):
        """
        Pickles the pandas dataframe so data doesn't have to be loaded again.
        """
        SC_tuple = (self.system,self._LabVIEW_labels,self._LabVIEW_indices,self._num_datasets,self._data)
        pickle.dump(SC_tuple, open(filename, "wb"))
        print('\nDataframe saved to '+filename)
        
    def LoadDataframe(self,filename):
        """
        Loads a previously pickled dataframe.
        """
        SC_tuple = pickle.load(open(filename, "rb"))
        (self.system,self._LabVIEW_labels,self._LabVIEW_indices,self._num_datasets,self._data) = SC_tuple
        print('\nDataframe loaded from '+filename)
        
    def MergeDatasets(self,second_object):
        """
        Merges two SlowControls objects, for instance if two were created with
        different column maps but need to be combined.
        """
        self._data = self._data.merge(second_object._data,how='outer')
        print('\nMerged data from two SlowControls objects.')
        
