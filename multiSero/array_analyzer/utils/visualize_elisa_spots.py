from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain


def visualize_elisa_spots(id2spot='./example_id2spots_AEPFeb3.csv', nrow=6, ncol=8, well='A1', measurements=None, export=None) -> object:
    '''
    Parameters
    ----------
    id2spot: str
                  Path to comma separated file.
    nrow : int
           Number of rows in the spot array.
    ncol : int
          Number of colums in the spot array.
    well : str
          Name of the well.
    measurements:
          N x nrow x ncol array which represents some measurement (e.g. OD).
    export

    Returns
    -------
    export: file
            Export the visualization to a file.


    '''

    id2spot_df=pd.read_csv(id2spot)
    spotarray = np.empty((nrow,ncol),dtype=object)
    for iID in np.arange(len(id2spot_df)):
        thisID=id2spot_df['ID'][iID]
        spotlist=id2spot_df['Spots'][iID]
        spotlist=spotlist.split(';')

        for iSpot in np.arange(len(spotlist)):
            tokens = spotlist[iSpot].split('-')
            rowidx = int(tokens[1]) - 1
            colidx = int(tokens[2]) - 1
            spotarray[rowidx,colidx]=thisID

    print(tabulate(spotarray))

    fig, ax = plt.subplots(nrow, ncol, sharex=True, sharey=True, gridspec_kw=dict(hspace=0, wspace=0),
                           subplot_kw=dict(xlim=[-1, 1], ylim=[-1, 1], xticks=[], yticks=[]))
    bbox_props = dict(boxstyle="circle,pad=0.3", fc='w', ec="b", lw=2)

    for rIdx in np.arange(nrow):
        for cIdx in np.arange(ncol):
            txt=ax[rIdx,cIdx].text(0,0,spotarray[rIdx,cIdx],ha='center',va='center', wrap=True,bbox=bbox_props)
            ax[rIdx,cIdx].axis('off')
           #TODO: set the size of circles to be uniform.

    plt.show()

    if measurements is not None:
        # TODO: overlay measurements.
        None

    if export is not None:
        # TODO: export as PNG.
        None

    return

if __name__ == '__main__':
    visualize_elisa_spots()