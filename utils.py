import pandas as pd
import numpy as np

# This function can be used to verify the sum of all the probabilities. 
def recursion_mult(df):
    
    # The purpose of this function is to do the element wise matrix multiplication using the recursion
    
    if df.empty:
        return
    else: 
        # 1. First remove the 1st column from the df, and save it as array
        ref_col = df.columns[0]
        ref_arr = df[ref_col].values
        df.drop(columns = ref_col, inplace = True)
        
        if df.empty:
            return ref_arr
        else:
            arr = recursion_mult(df)
            temp_arr = np.array([])
            for elem in arr:
                if len(temp_arr) == 0:
                    temp_arr = np.multiply(elem, ref_arr) 
                else:
                    temp_arr = np.concatenate((temp_arr, np.multiply(elem, ref_arr)))
                    
            return temp_arr
        
## Test recursion using the following code

'''
 
# Create a Dataframe
a,b,c = np.array([1,4,7]), np.array([2,5,8]), np.array([3,6,9])
df = pd.DataFrame([a,b,c], columns = ['A', 'B', 'C'])
print(df)

arr = recursion_mult(df) 
print('Sum of recursive multiplication is : {}'.format(arr.sum()) )

'''

# This function returns which column has the maximum number of bins
def max_binning(df):
    max_ = 0;
    for cols in df.columns:
        if len(df[cols].unique()) > max_:
            max_ = len(df[cols].unique())
    return max_


# Create Probability Matrix
def create_Prob_matrix(df, 
                      target_col_vector = []):
    '''
        Returns a Probability Matrix
        Input: Data frame, names of columns for which we need the P() matrix
        
    '''
    
    prob_matrix = [] 
    Prob_Terms_colname = []
    # for p in target_col_vector:
    #     new_col_name = 'P('+p+')'
    #     Prob_Terms.append(new_col_name)
    # Prob_Terms = ['P(RSRP_binned)', 'P(RSRQ_binned)', 'P(RSSI_binned)','P(PCC_PHY_Thruput_DL_binned)', 'P(PCC_SINR_binned)']

    for target_col in target_col_vector:
        P_col_name = 'P('+target_col+')'
        Prob_Terms_colname.append(P_col_name)
        print('Creating P() Col: {}'.format(P_col_name))
        
        # print('Col name: {}'.format(target_col))
        df = df.sort_values(by = target_col)
        prob_list = []
        for i in df[target_col].value_counts().values:
            # print(i)
            prob_list.append(i/len(df))

        prob_matrix.append(prob_list)    
        # print(prob_list)
        # prob_vec = np.array(prob_list)

    prob_matrix= np.array(prob_matrix)
    prob_matrix = prob_matrix.T

    prob_matrix = pd.DataFrame(prob_matrix)
    prob_matrix.columns = Prob_Terms_colname
    
    return prob_matrix

# Function for data binning 
def data_binning(df, binning_type = 1, bin_all_params = True):
    if bin_all_params :
        cols = df.columns.to_list()
        for col_name in cols:
            df.sort_values(by = col_name)
            arr = df[col_name].values
            new_colname = col_name + '_binned'
            # print('Working with col. name {}'.format(col_name))

            if binning_type == 0:
                df[new_colname] = iqd_binning(arr, 4)
            elif binning_type == 1:
                df[new_colname] = cut_binning(arr, 4)
            elif binning_type == 2:
                df[new_colname], _ = width_binning(arr,4)
    else:
        col_name = 'PCC_PHY_Thruput_DL'
        df.sort_values(by = col_name)
        arr = df[col_name].values

        if binning_type == 0:
            df[col_name + '_binned'] = iqd_binning(arr, 4)
        elif binning_type == 1:        
            df[col_name] = cut_binning(arr, 4)
        elif binning_type == 2:
            df[col_name + '_binned'], _ = width_binning(arr,4)


        col_name = 'PCC_SINR'
        df.sort_values(by = col_name)
        arr = df[col_name].values
        if binning_type == 0:
            df[col_name + '_binned'] = iqd_binning(arr, 4)
        elif binning_type == 1:
            df[col_name + '_binned'], _ = width_binning(arr,4)


        col_name = 'RSRP'
        df.sort_values(by = col_name)
        arr = df[col_name].values
        if binning_type == 0:
            df[col_name + '_binned'] = iqd_binning(arr, 4)
        elif binning_type == 1:
            df[col_name + '_binned'], _ = width_binning(arr,4)
            
    return df
 


# Binning data using pd.cut
def cut_binning(arr, N):
    '''
    N: Number of quantiles
    '''
    new_arr = pd.cut(arr, N, labels = False)
    return(new_arr)


def iqd_binning(arr, N):
    '''
    N: Number of quantiles
    '''
    new_arr = pd.qcut(arr, N, labels = False)
    return(new_arr)

def width_binning(arr, N = 4):
    '''
    N: Number of Stats / bins
    '''
    # W: width of the binning window
    
    new_arr = []
    # W = int((arr.max() - arr.min())/N)    
    W = int(np.ceil(len(arr)/N))
    arr.sort()
        
    for i in range(N):
        temp = [i]*W
        while bool(temp):        
            new_arr.append(temp.pop())
    
    return np.array(new_arr[0:len(arr)]), W