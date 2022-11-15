#!/usr/bin/env python
# coding: utf-8

# In[10]:


#!/usr/bin/env python3
"""

UVS Render - Ultraviolet-visible Spectra Render

"""
import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from abc import ABC, abstractmethod
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel, ExpressionModel
#ExponentialModel, LinearModel, VoigtModel, LorentzianModel

#abstract filter class
class uvs_filterer():
    #return data with filtered UV column
    #data must be formated like [[float(wavelength), float(UV)]]
    @abstractmethod
    def uvsfilter(self, data):
        pass

#crop 2D data array by x axis in the range from start to end
class uvs_filterer_cropx(uvs_filterer):
    #crop by x axis in the range of x from start to end
    x_start = 200
    x_end = 800
    
    def __init__(self, croprange=None):
        if isinstance(croprange, list) and len(croprange) == 2:
            self.x_start = croprange[0]
            self.x_end = croprange[1]
        
    #get only the range of the data
    def uvsfilter(self, data):
        start = np.where(x == self.x_start)
        end = np.where(x == self.x_end)
        #check wavelength points in the data array
        if start[0].size == 0 or end[0].size == 0:
            print("uvs_filterer_cropdata: Can't to find x_start (%d) or/and x_end (%d) in the data" % (x_start, x_end))
        else:
            #crop the data
            data = data[start[0][0]:end[0][0]]
        
        return data

    
#1D Baseline simple correction filter
class uvs_filterer_baselinesimple(uvs_filterer):
    #baseline wavelength, nm
    wavelength = 775
    
    #neighbourhood of baseline, nm
    neighbourhood = 24
    
    #set default wavelength
    def __init__(self, wavelength=None):
        if isinstance(wavelength, int) or isinstance(wavelength, float):
            self.wavelength = wavelength
        return
    
    #return data with filtered UV column
    #data must be formated like [[float(wavelength), float(UV)]]
    def uvsfilter(self, data):
        #find baseline point and neighbourhood indexes
        base = np.where(data[:,0] == self.wavelength)
        base_left = np.where(data[:,0] == self.wavelength-self.neighbourhood)
        base_right = np.where(data[:,0] == self.wavelength+self.neighbourhood)
        #get uv array
        uv = data[:,1]
        
        #check baseline points in the data array
        if base_left[0].size == 0 or base_right[0].size == 0 or base[0].size == 0:
            print("There are no baseline wavelength '%d nm' or it neighbourhood '%d nm' in the data array" % (self.wavelength, self.neighbourhood))
        else:
            #get average uv in the neighbourhood of baseline wavelength
            mean = np.mean(uv[base_left[0][0]:base_right[0][0]])
                  
            #strip UV data
            uv = uv - mean
        
        return np.vstack((data[:,0], uv)).T
    
#1D Savitzky–Golay filter
class uvs_filterer_savgol(uvs_filterer):
    #The length of the filter window (i.e., the number of coefficients)
    window_length = 50
    
    #The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    polyorder = 3
    
    #return data with filtered UV column
    #data must be formated like [[wavelength, UV]]
    def uvsfilter(self, data):
        uv = savgol_filter(data[:,1], self.window_length, self.polyorder)
        return np.vstack((data[:,0], uv)).T
        

#filter factory to render the UV data array
class uvs_filter(uvs_filterer):
    #filter chain
    filterers = []
    
    def __init__(self):
        self.filterers = []
    
    #add filter to chain: prepend or append
    def add_filterer(self, filterer, prepend=False):
        if not isinstance(filterer, uvs_filterer):
            return
        
        self.filterers = list(self.filterers)
        if prepend == True:
            self.filterers.insert(0, filterer)
        else:
            self.filterers.append(filterer)
    
    #return data array filtered through each filter in the chain (in the order of self.filterers)
    def uvsfilter(self, data):
        data_filtered = data
        for filt in self.filterers:
            data_filtered = filt.uvsfilter(data_filtered)
        return data_filtered
    

#parent class for UVS renderers
class uvs_renderer():
    #renderer name
    NAME = ""
    
    #source data without rendering as 2D array [[nm, uv]]
    source = [[]]
    
    #data to render as 2D array [[nm, uv]]
    data = [[]]
    
    #desirable wavelengths to calc UV (as 1D array, nm)
    #default 260 nm - DNA, 280 nm - protein
    wavelengths = [260, 280]
    
    #uvs_filter to render the data
    uvsfilter = None
    
    #precision of the results
    precision = 4
    
    #show results plot
    showplot = False
    
    #axis array of plot axis
    plotaxis = None
        
    #data (array) format: nm | uv
    def __init__(self):
        self.source = [[]]
        self.data = [[]]
        
        #add filter
        uvsfilter = uvs_filter()
        self.uvsfilter = uvsfilter
    
    #add data array, format: [[nm, uv]]
    def set_data(self, data):
        self.source = data
        self.data = data
        
    #add wavelengths list, format: [260, 280,...]
    def set_wavelengths(self, wavelengths):
        self.wavelengths = wavelengths
        
    #show or hide results plot
    def set_showplot(self, show):
        self.showplot = show
        
    def set_plotaxis(self, axis=None):
        self.plotaxis = axis
    
    #filter the data and calc UV
    def calcuv(self):
        data = self.data
        if self.uvsfilter != None:
            #filter the data
            data = self.uvsfilter.uvsfilter(data)
               
        return self.render(data)
    
    #calc UV for wavelengths
    #return 2D array [[nm -> int | uv -> float]], like [[260, 0.13], [280, 0.87]]
    @abstractmethod
    def render(self, data):
        pass
    
#calc UV by simplest method
#just filter data by savgol, strip baseline and get MAX UV-values in the neighbourhood of desirable wavelengths
class uvs_renderer_simple(uvs_renderer):
    #renderer name
    NAME = "simple"
    
    #neighbourhood for wavelengths, nm
    neighbourhood = 5
    
    #data (array) format: nm | uv
    def __init__(self, savgol=True, baselinesimple=True, baselinewave=None):
        super().__init__()

        uvsfilter = self.uvsfilter
        if savgol == True:
            uvsfilter.add_filterer(uvs_filterer_savgol())
            
        if baselinesimple == True:
            uvsfilter.add_filterer(uvs_filterer_baselinesimple(wavelength=baselinewave))
        
    #calc UV for wavelengths
    def render(self, data):
        uv = data[:,1]
        uv_arr = []
        for nm in self.wavelengths:
            #find wavelength neighbourhood indexes
            wavelength = np.where(data[:,0] == nm)
            wavelength_left = np.where(data[:,0] == nm-self.neighbourhood)
            wavelength_right = np.where(data[:,0] == nm+self.neighbourhood)
                  
            #check wavelength points in the data array
            if wavelength_left[0].size == 0 or wavelength_right[0].size == 0 or wavelength[0].size == 0:
                print("There are no wavelength '%d nm' or it neighbourhood '%d nm' in the data array" % (nm, self.neighbourhood))
            else:
                #find max UV
                uv_val = round(max(uv[wavelength_left[0][0]:wavelength_right[0][0]]), self.precision)
                uv_arr = np.append(uv_arr, uv_val)
        
        #create plot figure
        if self.showplot is True:
            uvsfilter = uvs_filter()
            uvsfilter.add_filterer(uvs_filterer_baselinesimple())
            source = uvsfilter.uvsfilter(self.source)
            self.plotaxis.plot(source[:,0], source[:,1], '.', markersize=1, c='black', label='data')
            self.plotaxis.plot(data[:,0], data[:,1], label='fit', c='r')
            
        return np.vstack((self.wavelengths, uv_arr)).T
    
#calc UV by gauss method
class uvs_renderer_gauss(uvs_renderer):
    #renderer name
    NAME = "gauss"
    
    #neighbourhood for wavelengths, nm
    neighbourhood = 5
    
    #default stdev as starting param for curve_fit
    default_stdev = 20
    
    #data (array) format: nm | uv
    def __init__(self, savgol=True, baselinesimple=True, baselinewave=None):
        super().__init__()

        uvsfilter = self.uvsfilter
        if savgol == True:
            uvsfilter.add_filterer(uvs_filterer_savgol())
            
        if baselinesimple == True:
            uvsfilter.add_filterer(uvs_filterer_baselinesimple(wavelength=baselinewave))
        
    #gauss function
    def gauss(self, x, amp, mean, stdev):
        return amp * np.exp(-(x - mean) ** 2.0 / (2.0 * stdev ** 2.0))
    
    #multiple gauss
    def gauss_multi(self, x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp = params[i]
            mean = params[i+1]
            stdev = params[i+2]
            y = y + self.gauss(x, amp=amp, mean=mean, stdev=stdev)
        return y
    
    #fit gauses for points
    def gauss_fit(self, x, y, points):
        params = []
        for point in points:
            params = np.append(params, [point[1], point[0], self.default_stdev])
            
        try:
            popt, pcov = curve_fit(self.gauss_multi, x, y, p0=params)
        except RuntimeError:
            print("Gauss fit - Optimal parameters not found. Initial parameters was used.")
            popt = params

        return popt

    #calc UV for wavelengths
    #return 2D array [[nm | uv]], like [[260, 0.13], [280, 0.87]]
    def render(self, data, sample='data'):
        uv = data[:,1]
        uv_amp = []
        for nm in self.wavelengths:
            #find wavelength neighbourhood indexes
            wavelength = np.where(data[:,0] == nm)
            wavelength_left = np.where(data[:,0] == nm-self.neighbourhood)
            wavelength_right = np.where(data[:,0] == nm+self.neighbourhood)
                  
            #check wavelength points in the data array
            if wavelength_left[0].size == 0 or wavelength_right[0].size == 0 or wavelength[0].size == 0:
                print("There are no wavelength '%d nm' or it neighbourhood '%d nm' in the data array" % (nm, self.neighbourhood))
            else:
                #find max UV
                uv_val = max(uv[wavelength_left[0][0]:wavelength_right[0][0]])
                uv_amp = np.append(uv_amp, uv_val)
        
        points = np.vstack((self.wavelengths, uv_amp)).T
        best_params = self.gauss_fit(data[:,0], uv, points)
        
        uv_arr = []
        i = 0
        for nm in self.wavelengths:
            uv_arr = np.append(uv_arr, round(best_params[3*i], self.precision))
            i += 1
        
        #create plot figure
        if self.showplot is True:
            #self.plotaxis.plot(data[:,0], data[:,1], '.', c='black', label=sample, markersize=1)
            uvsfilter = uvs_filter()
            uvsfilter.add_filterer(uvs_filterer_baselinesimple())
            source = uvsfilter.uvsfilter(self.source)
            self.plotaxis.plot(source[:,0], source[:,1], '.', markersize=1, c='black', label='data')
            self.plotaxis.plot(data[:,0], self.gauss_multi(data[:,0], *best_params), c='r', label='best fit', ls='-')
            
        return np.vstack((self.wavelengths, uv_arr)).T
    

#calc UV by lmfit library, include: Lorenz, Gauss, Exp
class uvs_renderer_lmfit(uvs_renderer):
    #renderer name
    NAME = "lmfit"
    
    #neighbourhood for wavelengths, nm
    neighbourhood = 15
    
    #default stdev as starting param for curve_fit
    default_stdev = 10
    
    #data (array) format: nm | uv
    def __init__(self, savgol=False, baselinesimple=True, baselinewave=None):
        super().__init__()

        uvsfilter = self.uvsfilter
        if savgol == True:
            uvsfilter.add_filterer(uvs_filterer_savgol())
            
        if baselinesimple == True:
            uvsfilter.add_filterer(uvs_filterer_baselinesimple(wavelength=baselinewave))
    
    #calc UV for wavelengths
    #return 2D array [[nm | uv]], like [[260, 0.13], [280, 0.87]]
    def render(self, data, sample='data'):
        x = data[:, 0]
        y = data[:, 1]
        #create model
        const_mod = ExpressionModel('const + 0*x')
        pars = const_mod.make_params(const=0)
        model = const_mod
        
        #add gauss components to model for each wavelengths
        for nm in self.wavelengths:
            #create gauss and add it to model
            index = 'g'+str(nm)+'_'
            gauss = GaussianModel(prefix=index)
            model = model + gauss
            
            #find maximum value in the neighbourhood of nm
            wavelength = np.where(x == nm)
            wavelength_left = np.where(x == nm-self.neighbourhood)
            wavelength_right = np.where(x == nm+self.neighbourhood)
            if wavelength_left[0].size == 0 or wavelength_right[0].size == 0 or wavelength[0].size == 0:
                print("There are no wavelength '%d nm' or it neighbourhood '%d nm' in the data array" % (nm, self.neighbourhood))
                uv_max = 2
            else:
                #find max UV
                uv_max = max(y[wavelength_left[0][0]:wavelength_right[0][0]])
            
            #set params
            pars.update(gauss.make_params())
            pars[index+'center'].set(value=nm, min=nm-self.neighbourhood, max=nm+self.neighbourhood)
            pars[index+'sigma'].set(value=self.default_stdev, min=5)
            pars[index+'amplitude'].set(value=uv_max, min=0.02)
        
        #start fitting and get best params
        init = model.eval(pars, x=x)
        out = model.fit(y, pars, x=x)
        best_params = out.best_values
        
        #create plot figure
        if self.showplot is True:
            comps = out.eval_components(x=x)
            self.plotaxis.plot(x, y, '.', c="black", label=sample, markersize=1)
            self.plotaxis.plot(x, out.best_fit, ls='-', c="r", label='best fit')
            for nm in self.wavelengths:
                index = 'g'+str(nm)+'_'
                self.plotaxis.plot(x, comps[index], ls='--', label='component '+str(nm)+'nm')
            
        uv_arr = []
        i = 0
        for nm in self.wavelengths:
            index = 'g'+str(nm)+'_'
            uv_arr = np.append(uv_arr, round(best_params[index+'height'], self.precision))
            i += 1
        
        return np.vstack((self.wavelengths, uv_arr)).T
    
    
#default plot settings for UV spectra
class uvs_plot():
    #Figure size
    width = 14
    height = 10
    
    #axes titles
    title_x = 'Wavelength, nm'
    title_y = 'Absorbance, AU'
    
    #fig
    fig = None
    
    #axes
    axes = None
    
    def plot_create(self, axescount=1, title=""):
        fig, axes = plt.subplots(1, axescount)
        
        fig.suptitle(title)
        fig.set_figwidth(self.width)
        fig.set_figheight(self.height/axescount)
        
        if axescount == 1:
            axes = np.array([axes])
            
        for ax in axes:
            ax.axhline(0, color='black', lw=1)
            ax.set_xlabel(self.title_x)
            ax.set_ylabel(self.title_y)
        
        self.fig = fig
        self.axes = axes
        return axes
    
    def plot_show(self):
        for ax in self.axes:
            ax.legend(loc="best")
            
        plt.show()
        
        
class uvs_render():
    #source data as 2D array
    source = [[]]
    
    #list of uvs_renderers
    renderers = [];
    
    #desirable wavelengths to calc UV (as 1D array, nm)
    #default 260 nm - DNA, 280 nm - protein
    wavelengths = [260, 280]
    
    #define default values
    def __init__(self):
        self.source = [[]]
        self.renderers = []
    
    #add renderer to list of renderers
    def add_renderer(self, renderer, prepend=False):
        if not isinstance(renderer, uvs_renderer):
            return
        
        self.renderers = list(self.renderers)
        if prepend == True:
            self.renderers.insert(0, renderer)
        else:
            self.renderers.append(renderer)
            
    #file -> path to file (array type like .csv)
    #file format -> [[wave (nm), uv for sample1, uv for sample2, ...]]
    #and first line as titles -> example: [[wavelength (nm), Blank, Buffer, Sample-1, Sample-2]]
    def get_data_from_csv(self, file=None, delimiter=';'):
        source = [[]]
        if isinstance(file, str):
            try:
                #get data from file
                source = np.genfromtxt(file, delimiter=delimiter, dtype=None, encoding=None)
                #remove d-quotes for string values
                source = np.char.strip(source, '"')
                #replace comma to dot, needed if float values with comma
                source = np.char.replace(source,",",".")
            except ValueError:
                print("Can't to read file.")
        self.source = source
    
    #get only the range of the data
    def crop_data(self, x_start, x_end):
        source = self.source
        x = source[:,0].astype(str)
        start = np.where(x == str(x_start))
        end = np.where(x == str(x_end))
        #check wavelength points in the data array
        if start[0].size == 0 or end[0].size == 0:
            print("Can't to find x_start (%d) or/and x_end (%d) in the data" % (x_start, x_end))
        else:
            #crop the data
            titles = source[0:1]
            source = np.concatenate((titles, source[start[0][0]:end[0][0]]), axis=0)
        
        self.source = source
                
    #get list of titles. Used as headers in the resulting 2D array
    def get_titles_list(self):
        #format - [[sample title, UV at X nm calculated by renderer 1, UV at Y nm by renderer 1, UV at X nm by renderer 2, UV at Y nm by renderer 2, ...]]
        titles_arr = ['Sample']
        for renderer in self.renderers:
            for wave in self.wavelengths:            
                title = str(wave) + " (" + str(renderer.NAME) + ")"
                titles_arr = np.append(titles_arr, [title])
        return titles_arr
        
    #save csv file with results
    def save_csv(self, file, results):
        try:
            np.savetxt(file, results, delimiter=";", fmt='%s')
            print("Results saved successfully!")
        except ValueError:
            print("Can't to save results. Please check the directory and write permissions for filepath.")
            
        
    #render data and find UV at desirable wavelengths
    #show plots - show plots here to estimate curves quality
    #save_file - False OR path to csv file to save results
    def render(self, showplot=False, save_file=False):
        source = self.source
        #get sample titles from the data first line
        cols = source[0,:]

        #get working array without titles and only with float values
        data = np.delete(source,(0),axis=0)
        data = data.astype(float)
        
        #get first col as x axis
        x = data[:,0]

        #get only samples uv-values (without wavelengths)
        data_uv = np.transpose(data)
        data_uv = np.delete(data_uv,(0),axis=0)
        
        #create resulting array
        result_arr = np.array([self.get_titles_list()])
        
        #get data from each sample, render it and calc UV maximum
        col = 1
        for val in data_uv:
            if showplot == True:
                uvsplot = uvs_plot()
                axes = uvsplot.plot_create(axescount=len(self.renderers), title=cols[col])
            
            data_arr = np.transpose([x, val])
            sample_arr =  np.array([cols[col]])
            nrend=0 #num of renderer
            for renderer in self.renderers:
                renderer.set_data(data_arr)
                renderer.set_wavelengths(self.wavelengths)
                renderer.set_showplot(showplot)
                if showplot == True:
                    axes[nrend].set_title(renderer.NAME)
                    renderer.set_plotaxis(axes[nrend])
                uv_res = renderer.calcuv()
                sample_arr = np.append(sample_arr, [uv_res[:,1]])
                nrend += 1
            
            #show plot
            if showplot == True:
                uvsplot.plot_show()
            
            #add to resulting array
            result_arr = np.append(result_arr, [sample_arr], axis=0)
            
            #go to the next sample
            col += 1
        
        if isinstance(save_file, str):
            self.save_csv(save_file, result_arr)
            
        return result_arr

#check params from bash
params = sys.argv
if len(sys.argv) > 0 and sys.argv[0] == 'uvs_render.py':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to csv file with source data. Example: '/home/inputData.csv'")
    parser.add_argument("--output", help="Path to csv file with output results data. Example: '/home/results.csv'")
    parser.add_argument("--delim", help="Delimiter for input csv file. Default: --delim=';'", default=';')
    parser.add_argument("--waves", help="Desirable wavelengths for finding a max(UV). Default: --waves='260,280'", default='260,280')
    parser.add_argument("--crop", help="Crop input data in the range of the desirable wavelengths (this procedure can help for fit the gausses). Default: --crop='200-800'", default='200-800')
    parser.add_argument("--baselinewave", help="Baseline wavelenght in nm. Default: --baselinewave=775", default=775)
    args=parser.parse_args()

    try:
        source_file = args.input
        results_file = args.output
        delim = args.delim
        waves = np.fromstring(args.waves, dtype=int, sep=',')
        crop = np.fromstring(args.crop, dtype=int, sep='-')
        baselinewave = args.baselinewave
        showplot = False
    except IndexError:
        print("The --input and --output parameters are required")
        print("Example: python3 uvs_render.py --input '/home/inputData.csv' --output '/home/results.csv'")
else:
    #source_file = "/home/goncharuk/Downloads/python/2022-10-25-k2-rm1234[buf].csv"
    source_file = "/home/test.csv"
    results_file = "/home/test_results.csv"
    delim = ";"
    waves = [380,540]
    crop = [310,820]
    baselinewave = None
    showplot = True

render = uvs_render()
render.get_data_from_csv(source_file, delim)
render.crop_data(x_start=crop[0], x_end=crop[1])
render.add_renderer(uvs_renderer_simple(baselinewave=baselinewave))
render.add_renderer(uvs_renderer_gauss(baselinewave=baselinewave))
render.add_renderer(uvs_renderer_lmfit(baselinewave=baselinewave))
render.wavelengths = waves
render.render(showplot=showplot, save_file=results_file)
print("Done! Thank you for choosing us! Good Luck!")


# In[ ]:




