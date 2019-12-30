import sympy as sp
import numpy as np
import pandas as pd
# Y=np.load("../data/Y.npy")
from .YData import Y
from .DATA_DM2 import data_PM,data_AVR,data_G
from .PF import Power_flow
from .A_matrix3 import A_Generation
import matplotlib.pyplot as plt
np.set_printoptions(precision=6, suppress=True)

Load_96_ori=np.load("data/load_96.npy")


headers=['index','Node_type','U','delta','P_L','Q_L','P_G','Q_G','X','B']
Bus_data=pd.read_csv('data/Bus_data.csv',header=None)
Bus_data.columns = headers
bus_data=Bus_data.astype(float)

num_Node=bus_data.shape[0]
num_level=87

# 求特征根及特征向量
def get_UV(A):
    """
    获得A矩阵左右特征向量
    *** 特征向量矩阵中特征向量均为列向量 ***
    """
    vals, U = np.linalg.eig(A)
    V=np.linalg.inv(U).T.copy()
    return vals,U,V

# 取初始点为最大值点
index_max = np.argwhere(Load_96_ori == max(Load_96_ori))[0][0]
Load_96 = np.zeros(Load_96_ori.shape[0])
Load_96[:Load_96_ori.shape[0] - index_max] = Load_96_ori[index_max:]
Load_96[Load_96_ori.shape[0] - index_max:] = Load_96_ori[:index_max]

P_L_index = bus_data['index'][bus_data['P_L'].values != 0].values
P_L_values = bus_data['P_L'][bus_data['P_L'].values != 0].values

# Q_L_index = bus_data['index'][bus_data['Q_L'].values != 0].values
# Q_L_values = bus_data['Q_L'][bus_data['Q_L'].values != 0].values

Load_96_nor=Load_96.copy()/Load_96[0]  #归一化


def Eigenval(t):

    P_L_valuesi=P_L_values*Load_96_nor[t]

    #计算96个运行时刻的A矩阵及特征根
    bus_data_i = bus_data.copy()

    bus_data_i["P_L"][P_L_index - 1] = P_L_valuesi
    # bus_data_i["P_G"][P_G_index - 1] = P_G_values_arr[i,]
    # bus_data_i["Q_L"][Q_L_index - 1] = Q_L_values_arr[i,]
    A_mat_i=A_Generation(Power_flow,Y,bus_data_i,data_G,data_AVR,data_PM)
    vals=get_UV(A_mat_i)[0]

    return vals

# print(Eigenval(1))