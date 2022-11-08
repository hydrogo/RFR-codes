import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def baseflow_separation(strflow, f1 = 0.925, figure=False):
    '''
    requirements: 
    import numpy as np 
    import pandas as pd
    
    Input: 
    pandas timeseries of streamflow with datetime index (for appropriate figure axis)
    
    Output:
    pandas dataframe with streamflow and 3 timeseries of baseflow
    
    f1 parameter: 
    usually near 0.92-0.95
    '''
    # set initial conditions
    f2 = (1. + f1) / 2.
    sumd = len(strflow)

    # init surface flow
    surfq = np.zeros(sumd)
    # set surfq first value to 0.5 of strflow
    surfq[0] = strflow[0] * .5

    # init passes
    baseq_1 = np.zeros(sumd)
    baseq_2 = np.zeros(sumd)
    baseq_3 = np.zeros(sumd)

    ### first (forward) pass
    # init first calculation step
    baseq_1[0] = strflow[0] - surfq[0]
    
    for i in range(1, sumd):
        surfq[i] = f1 * surfq[i-1] + f2 * (strflow[i] - strflow[i-1])
        if surfq[i] < 0: 
            surfq[i] = 0

        baseq_1[i] = strflow[i] - surfq[i]

        if baseq_1[i] < 0: 
            baseq_1[i] = 0
        if baseq_1[i] > strflow[i]: 
            baseq_1[i] = strflow[i]
    
    ### second (backward) pass
    # init first calculation step
    baseq_2[sumd-1] = baseq_1[sumd-1]

    for i in range(sumd-2, -1, -1):
        surfq[i] = f1 * surfq[i+1] + f2 * (baseq_1[i] - baseq_1[i+1])
        if surfq[i] < 0: 
            surfq[i] = 0

        baseq_2[i] = baseq_1[i] - surfq[i]

        if baseq_2[i] < 0: 
            baseq_2[i] = 0
        if baseq_2[i] > baseq_1[i]: 
            baseq_2[i] = baseq_1[i]
        
    ### third (forward) pass
    # init first calculation step
    baseq_3[sumd-1] = baseq_1[sumd-1]

    for i in range(1, sumd):
        surfq[i] = f1 * surfq[i-1] + f2 * (baseq_2[i] - baseq_2[i-1])

        if surfq[i] < 0: 
            surfq[i] = 0

        baseq_3[i] = baseq_2[i] - surfq[i]

        if baseq_3[i] < 0: 
            baseq_3[i] = 0
        if baseq_3[i] > baseq_2[i]: 
            baseq_3[i] = baseq_2[i]
    
    ### wrap up everything in one place
    result = pd.DataFrame({'Streamflow': strflow, 
                    'Baseflow_1': baseq_1, 
                    'Baseflow_2': baseq_2, 
                    'Baseflow_3': baseq_3}, index=strflow.index)
    
    if figure == True:
        return result, result.plot(figsize=(15,10), style=['--g', '--b', '-k', '-r'])
    else:
        return result
