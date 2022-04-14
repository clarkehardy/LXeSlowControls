import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import matplotlib.dates as mdates
import pandas as pd
import time
from tqdm import tqdm, trange



class SlowControls(object):
    def __init__(self, datasets, system = None, labels = None, indices = None):
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
        if labels==None:
            first_name = datasets[0].split('/')[1]
            system = first_name[0:-20]
            if system=='SS':
                self.__LabVIEW_labels = ['date','T_in','T_out','T_Cu_top','T_Cu_bottom','T_cell_mid','P_XP3','P_XP5','P_CCG','cool_on']
                self.__LabVIEW_indices = np.array([0,8,7,5,1,3,22,23,24,35])
            elif system=='LS':
                self.__LabVIEW_labels = ['date','T_Cu_bottom','T_cell_top','T_cell_mid','T_cell_bottom','T_Cu_top','T_ambient','T_in','T_top_flange','T_set_min','T_set_max','mass_flow','P_XP3','P_XP5','P_CCG','cool_on','water_flow','discharge_pressure','suction_pressure','coil_in_temp','coil_out_temp','water_temp','fridge_CT','coil2_out_temp']
                self.__LabVIEW_indices = np.array([0,1,2,3,4,5,6,7,8,10,11,21,22,23,24,35,38,39,40,41,42,43,44,45])
            elif system=='Purifier':
                self.__LabVIEW_labels = ['date','set_temp','heater_on','T_Al_middle','T_Al_top','T_Al_bottom','T_upper','T_top_flange','T_lower','T_bottom_flange']
                self.__LabVIEW_indices = np.array([0,1,2,3,4,5,6,7,8,9])

        else:
            self.__LabVIEW_labels = labels
            self.__LabVIEW_indices = indices

        self.system = system #label of the system, like "SS", "LS", "Purifier" (and typically file prefix of datafiles)
        datasets, = self.__FormatArgs((datasets,))
        self.__num_datasets = len(datasets)
        points = []
        cols = 0
        for i in range(self.__num_datasets):
            print('\nOpening dataset in {}...'.format(datasets[i]))
            with open(datasets[i],'r') as infile:
                reader = csv.reader(infile,delimiter='\t')
                for line in reader:
                    old_cols = cols
                    cols = len(line)
                    if cols!=old_cols:
                        print('File '+datasets[i]+' has {} columns'.format(cols))

                looper = tqdm(reader, desc="Processing lines...")
                for line in looper:
                    points.append(pd.DataFrame([np.array(line)[self.__LabVIEW_indices].astype(float)],
                                           columns=self.__LabVIEW_labels,index=[str(i+1)]))

        


        self.__data = pd.concat(points)
        self.__data['date'] = self.__data['date']-2082844800
        if('P_CCG' in self.__data):
            self.__data['P_CCG'] = self.__data['P_CCG']*1e-6 #i don't like this, can we change somehow? 

        min_date = min(self.__data['date'])
        max_date = max(self.__data['date'])
        print('\nFound {} readings between {} and {}.'.format(len(self.__data),\
                                                           datetime.fromtimestamp(min_date).strftime('%m/%d/%Y %H:%M:%S'),\
                                                           datetime.fromtimestamp(max_date).strftime('%m/%d/%Y %H:%M:%S')))

        print("Sorting by date")
        self.__data = self.__data.sort_values("date")
        print('\nCreated pandas dataframe containing slow controls data.')
        print('\nTime: {:.3f} seconds.\n'.format(time.time()-start))
        
    def __GetDateInterval(self,start_date,end_date):
        """
        Returns a slice that can be used to index for
        a given range of dates.
        """
        start_date,end_date = self.__FormatArgs((start_date,end_date))
        ranges = []
        for i in range(len(start_date)):
            start = datetime.timestamp(start_date[i])
            end = datetime.timestamp(end_date[i])
            dates = self.__data['date']
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
            
    def __GetHoursFromTime(self,start_time,end_time):
        """
        Returns a list containing arrays of the number
        of hours since a specified starting time.
        """
        start_time,end_time = self.__FormatArgs((start_time,end_time))
        ranges = self.__GetDateInterval(start_time,end_time)
        hours_array = []
        for i in range(len(start_time)):
            secs_from_start = np.array(self.__data['date'][ranges[i]])-self.__data['date'][ranges[i].start]
            hours_array.append(secs_from_start/3600.)
        return hours_array,ranges

    #returns labels and indices
    def GetLabels(self):
        return self.__LabVIEW_labels, self.__LabVIEW_indices


    def PlotVsDate(self,quantity,start_date,end_date,labels=[],title='',ylabel='',semilogy=False):
        """
        Method for plotting a given quantity or set of quantities
        over a specified date range.
        """
        if(quantity == []):
            fig, ax = plt.subplots()
            return fig, ax 
            
        quantity, = self.__FormatArgs((quantity,))
        if labels==[] and type(start_date)==list:
            labels = list(range(1,len(start_date)+1))
            labels = list(map(str,labels))
        else:
            labels = ['']
        start_date,end_date,labels = self.__FormatArgs((start_date,end_date,labels))
        ranges = self.__GetDateInterval(start_date,end_date)
        if ylabel=='':
            ylabel = quantity[0]
        fig,ax = plt.subplots()
        for i in range(len(start_date)):
            dates = self.__data['date'][ranges[i]]
            dates = [datetime.fromtimestamp(date) for date in dates]
            for j in range(len(quantity)):
                yvals = self.__data[quantity[j]][ranges[i]]
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
        quantity, = self.__FormatArgs((quantity,))
        if labels==[] and type(start_date)==list:
            labels = list(range(1,len(start_date)+1))
            labels = list(map(str,labels))
        else:
            labels = ['']
        start_date,end_date,labels = self.__FormatArgs((start_date,end_date,labels))
        hours,ranges = self.__GetHoursFromTime(start_date,end_date)
        if ylabel=='':
            ylabel = quantity[0]
        fig,ax = plt.subplots()
        for i in range(len(start_date)):
            for j in range(len(quantity)):
                yvals = self.__data[quantity[j]][ranges[i]]
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
        ranges = self.__GetDateInterval(start_date,end_date)
        dates = self.__data['date'][ranges[0]]
        dates = [datetime.fromtimestamp(date) for date in dates]
        return np.array(dates)

    def GetHoursArray(self,start_date,end_date):
        """
        Returns an array of the hours since the start time between
        a given start time and end time.
        """
        hours,ranges = self.__GetHoursFromTime(start_date,end_date)
        return hours[0]

    def GetQuantityArray(self,quantity,start_date,end_date):
        """
        Returns an array of a specified quantity between a given
        start date and end date.
        """
        ranges = self.__GetDateInterval(start_date,end_date)
        return np.array(self.__data[quantity][ranges[0]])
    
    def __FormatArgs(self,args):
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
        for i in range(len(self.__LabVIEW_labels)):
            print('index: {}, quantity: {}'.format(self.__LabVIEW_indices[i],str(self.__LabVIEW_labels[i])))
    
    def PrintHead(self):
        """
        Prints the pandas DataFrame head.
        """
        print(self.__data.head())

    def PrintTimeBounds(self):
        dates = self.__data['date']
        print("From ",end='')
        print(datetime.fromtimestamp(dates.min()), end='')
        print(" to ", end='')
        print(datetime.fromtimestamp(dates.max()))

    def GetTimeBounds(self):
        dates = self.__data['date']
        return datetime.fromtimestamp(dates[0]), datetime.fromtimestamp(dates[-1])

    #gets the most recent values for each process variable (quantity)
    def GetMostRecentValues(self):
        label_val_tuples = [] #[[label, val], [label, val], ...]
        for l in self.__LabVIEW_labels:
            label_val_tuples.append([l, self.__data[l][-1]])

        return label_val_tuples


