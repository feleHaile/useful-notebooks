import numpy as np 
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import subprocess



def color_legend_texts(leg):
    """Color legend texts based on color of corresponding lines"""
    for line, txt in zip(leg.get_lines(), leg.get_texts()):
        txt.set_color(line.get_color())
    for item in leg.legendHandles:
        item.set_visible(False)
        
def legend_texts_sort(ax):
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: len(str(t[0]))))
    #ax.legend(handles, labels)
    #leg=ax.legend(handles, labels, bbox_to_anchor=(0.7, 1.15),frameon=False,markerscale=0)
    return handles,labels
    
def sort_color_legend(ax1,ax2,kw):
    handles,labels=legend_texts_sort(ax1)
    leg=ax2.legend(handles, labels,frameon=False,markerscale=0,**kw)
    color_legend_texts(leg)
    return ax2
    
def color_legend(ax = None, props = {'weight':'bold', 'size':17}):
    handles, labels  = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc=4, frameon=False, prop=props)
    color_legend_texts(leg)  
    
def logformat(y,pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)
    
def gen_fig(nrows,ncols,width=None,aspect=None,pad=None,kw=None,grid=None):
    if not width:
        width=10
    if not aspect:
        aspect=0.5
    if not pad:
        pad=dict(wspace=0, hspace=0,left=0.06, bottom=0.09, right=0.9, top=0.99)
    if not kw:
        kw=dict(sharey='row', sharex='col')
    fig, axes = plt.subplots(nrows,ncols,figsize=(width,width*aspect),gridspec_kw=grid, **kw)
    fig.subplots_adjust(**pad)
    return fig,axes 

def animation(ins,outs,outfile,indir):
    try:
        os.makedirs(outs)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    command='convert -delay 60 -loop 0'+' '+ins+' '+outfile+'; rm -rf '+indir
    subprocess.call(command,shell=True)
    
def gif2mp4(infile,outs,outfile):
    try:
        os.makedirs(outs)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    command='ffmpeg -f gif -i'+' '+infile+' '+outfile
    subprocess.call(command,shell=True) 
 

def move(infile,outfile):
    command='mv'+' '+infile+' '+outfile
    subprocess.call(command,shell=True)

def copy(infile,outfile):
    command='cp'+' '+infile+' '+outfile
    subprocess.call(command,shell=True)
    
def make_dir(folder): 
    try:
        os.makedirs(folder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def write(temp,line):
    foo=open(temp,'a')
    foo.write(line+'\n')
    foo.close()

def figsave(fig,filename,basefolder):  
    fig.savefig(basefolder+'eps/'+filename+'.eps')
    fig.savefig(basefolder+'pdf/'+filename+'.pdf')
    fig.savefig(basefolder+'png/'+filename+'.png')
    
def lsqfity(X, Y):
    """
    Calculate a "MODEL-1" least squares fit.

    The line is fit by MINIMIZING the residuals in Y only.

    The equation of the line is:     Y = my * X + by.
    
    input and output as follows:

    my, by, ry, smy, sby = lsqfity(X,Y)
    X     =    x data (vector)
    Y     =    y data (vector)
    my    =    slope
    by    =    y-intercept
    ry    =    correlation coefficient
    smy   =    standard deviation of the slope
    sby   =    standard deviation of the y-intercept

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Determine the size of the vector.
    n = len(X)
    
    # Calculate the sums.

    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sx2 = np.sum(X ** 2)
    Sxy = np.sum(X * Y)
    Sy2 = np.sum(Y ** 2)

    # Calculate re-used expressions.
    num = n * Sxy - Sx * Sy
    den = n * Sx2 - Sx ** 2

    # Calculate my, by, ry, s2, smy and sby.
    my = num / den
    by = (Sx2 * Sy - Sx * Sxy) / den
    ry = num / (np.sqrt(den) * np.sqrt(n * Sy2 - Sy ** 2))

    diff = Y - by - my * X

    s2 = np.sum(diff * diff) / (n - 2)
    smy = np.sqrt(n * s2 / den)
    sby = np.sqrt(Sx2 * s2 / den)

    return my, by, ry, smy, sby   
    
def linestyle():
	linestyles = OrderedDict(
    		[('solid',               (0, ())),
	     	('loosely dotted',      (0, (1, 10))),
     		('dotted',              (0, (1, 5))),
     		('densely dotted',      (0, (1, 1))),

     		('loosely dashed',      (0, (5, 10))),
     		('dashed',              (0, (5, 5))),
     		('densely dashed',      (0, (5, 1))),

     		('loosely dashdotted',  (0, (3, 10, 1, 10))),
     		('dashdotted',          (0, (3, 5, 1, 5))),
     		('densely dashdotted',  (0, (3, 1, 1, 1))),

     		('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     		('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     		('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
