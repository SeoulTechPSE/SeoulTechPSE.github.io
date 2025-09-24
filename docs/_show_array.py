import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def show_array(shape, sel, filename=None):
    """
    Visualize indexing of arrays
    """
    
    data = np.zeros(shape)
    exec(f'data[{sel}] = 1')
    
    fig, ax = plt.subplots(1, 1, figsize=shape)
    ax.set_frame_on(False)

    ax.patch.set_facecolor('white')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    size = 0.96
    for (m, n), w in np.ndenumerate(data):
        color = '#1199ff' if w > 0 else '#eeeeee'
        rect = plt.Rectangle([n -size/2, m -size/2], 
                             size, size, 
                             facecolor=color,
                             edgecolor=color)
        ax.add_patch(rect)
        ax.text(n, m, f'({m}, {n})', ha='center',
                                     va='center',
                                     fontsize=12)
    ax.autoscale_view()
    ax.invert_yaxis()
    
    if sel ==':, :':
        ax.set_title('data\n', fontsize=12)
    else:
        ax.set_title(f'data[{sel}]\n', fontsize=12)
        
    if filename:
        fig.savefig(filename + ".png", dpi=200)
        fig.savefig(filename + ".svg")
        fig.savefig(filename + ".pdf")


def show_array_broadcasting(a, b, filename=None):
    """
    Visualize broadcasting of arrays
    """
 
    colors = ['#1199ff', '#eeeeee']
    
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.7))

    # -- a --
    data = a
    
    ax = axes[0]
    ax.set_frame_on(False)    
    ax.patch.set_facecolor('white')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    size = 0.96
    color = colors[0]
    for (m, n), w in np.ndenumerate(data):

        rect = plt.Rectangle([n -size/2, m -size/2],
                             size, size,
                             facecolor=color,
                             edgecolor=color)
        ax.add_patch(rect)
        ax.text(n, m, f'{data[m, n]}', ha='center', va='center', fontsize=14)        

    ax.text(3, 1, "+", ha='center', va='center', fontsize=22)        
    ax.autoscale_view()
    ax.invert_yaxis()

    # -- b --
    data = np.zeros_like(a) + b

    ax = axes[1]
    ax.set_frame_on(False)     
    ax.patch.set_facecolor('white')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    size = 0.96
    for (m, n), w in np.ndenumerate(data):
        
        if (np.argmax(b.T.shape) == 0 and m == 0) or (np.argmax(b.T.shape) == 1 and n == 0):
            color = colors[0]
        else:
            color = colors[1]
            
        rect = plt.Rectangle([n -size/2, m -size/2],
                             size, size,
                             facecolor=color,
                             edgecolor=color)
        ax.add_patch(rect)

        ax.text(m, n, f'{data[n, m]}', ha='center', va='center', fontsize=14)        

    ax.text(3, 1, "=", ha='center', va='center', fontsize=22)        
    ax.autoscale_view()
    ax.invert_yaxis()

    # -- c --
    data = a + b
    
    ax = axes[2]
    ax.set_frame_on(False) 
    ax.patch.set_facecolor('white')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
  
    size = 0.96
    color = colors[0]
    for (m, n), w in np.ndenumerate(data):

        rect = plt.Rectangle([n -size/2, m -size/2],
                             size, size,
                             facecolor=color,
                             edgecolor=color)
        ax.add_patch(rect)
        
        ax.text(m, n, f'{data[n, m]}', ha='center', va='center', fontsize=14)        

    ax.autoscale_view()
    ax.invert_yaxis()
    
    #fig.tight_layout()
        
    if filename:
        fig.savefig(filename + ".png", dpi=200)
        fig.savefig(filename + ".svg")
        fig.savefig(filename + ".pdf")        



def show_array_aggregation(data, axis, filename=None):
    """
    Visualize aggregation of arrays
    """

    colors = ['#1199ff', '#ee3311', '#66ff22']
    
    fig, axes = plt.subplots(2, 1, figsize=(3, 6))
    
    # -- data --
    ax = axes[0]
    ax.set_frame_on(False)
    ax.patch.set_facecolor('white')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
       
    size = 0.96
    for (m, n), w in np.ndenumerate(data):

        if axis is None:
            color = colors[0]
        elif axis == 1:
            color = colors[m]
        else:
            color = colors[n]
            
        rect = plt.Rectangle([n -size/2, m -size/2],
                             size, size,
                             facecolor=color,
                             edgecolor=color)
        ax.add_patch(rect)

        ax.text(n, m, f'{data[m, n]}', ha='center', va='center', fontsize=14)
        
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_title("data", fontsize=12)

    # -- data aggregation -- 
    
    if axis is None:
        adata = np.atleast_2d(data.sum())
    elif axis == 0:
        adata = data.sum(axis=axis)[:, np.newaxis]
    else:
        adata = data.sum(axis=axis)[:, np.newaxis]     
   
    ax = axes[1]
    ax.set_frame_on(False)
    ax.patch.set_facecolor('white')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    size = 1.0
    for (m, n), w in np.ndenumerate(data):
        color = 'white'
        rect = plt.Rectangle([n -size/2, m -size/2],
                         size, size,
                         facecolor=color,
                         edgecolor=color)
        ax.add_patch(rect)        
    
    size = 0.96
    for (m, n), w in np.ndenumerate(adata):

        if axis is None:
            color = colors[0] 
            rect = plt.Rectangle([1 +m -size/2, n -size/2],
                         size, size,
                         facecolor=color,
                         edgecolor=color)
            ax.add_patch(rect)
            
            ax.text(1 +m, n, f'{adata[m, n]}', ha='center', va='center', fontsize=14)
            
        if axis == 0:
            color = colors[m]
            rect = plt.Rectangle([m -size/2, n -size/2],
                                 size, size,
                                 facecolor=color,
                                 edgecolor=color)
            ax.add_patch(rect)
     
            ax.text(m, n, f'{adata[m, n]}', ha='center', va='center', fontsize=14)
        
        if axis == 1:
            color = colors[m]
            rect = plt.Rectangle([1 +n -size/2, m -size/2],
                                 size, size,
                                 facecolor=color,
                                 edgecolor=color)
            ax.add_patch(rect)
     
            ax.text(1 +n, m, f'{adata[m, n]}', ha='center', va='center', fontsize=14)        

    ax.autoscale_view()
    ax.invert_yaxis()
    
    if axis is not None:
        ax.set_title(f'data.sum(axis={axis})', fontsize=12)
    else:
        ax.set_title('data.sum()', fontsize=12)
    
    #fig.tight_layout()
    
    if filename:
        fig.savefig(filename + ".png", dpi=200)
        fig.savefig(filename + ".svg")
        fig.savefig(filename + ".pdf")