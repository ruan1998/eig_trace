import sympy as sp
import numpy as np
import pandas as pd
from .YData import Y

from DATA_DM2 import data_PM,data_AVR,data_G
from PF import Power_flow
from A_matrix3 import A_Generation
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)
Load_96_ori=np.load("data/load_96.npy")
headers=['index','Node_type','U','delta','P_L','Q_L','P_G','Q_G','X','B']
Bus_data=pd.read_csv('data/Bus_data.csv',header=None)
Bus_data.columns = headers
bus_data=Bus_data.astype(float)

num_Node=bus_data.shape[0]

# 按节点类型分类，并保留原序号
index_bal = bus_data['index'][bus_data['Node_type'] == 3].values.astype(int)
index_PV = bus_data['index'][bus_data['Node_type'] == 2].values.astype(int)
index_PQ = np.sort(np.insert(bus_data['index'][bus_data['Node_type'] == 1].values, 0,
                             bus_data['index'][bus_data['Node_type'] == 0].values, 0).astype(int))


num_rPoint=96
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

def Eigenval_all(Load_96_ori,bus_data):
    # 取初始点为最大值点
    index_max = np.argwhere(Load_96_ori == max(Load_96_ori))[0][0]
    Load_96 = np.zeros(Load_96_ori.shape[0])
    Load_96[:Load_96_ori.shape[0] - index_max] = Load_96_ori[index_max:]
    Load_96[Load_96_ori.shape[0] - index_max:] = Load_96_ori[:index_max]
    # for i in range(Load_96_ori.shape[0]):
    #     Load_96[i]=Load_96_ori[95-i]

    P_L_index = bus_data['index'][bus_data['P_L'].values != 0].values
    P_L_values = bus_data['P_L'][bus_data['P_L'].values != 0].values
    # P_G_index = bus_data['index'][bus_data['P_G'].values != 0].values
    # P_G_values = bus_data['P_G'][bus_data['P_G'].values != 0].values
    # Q_L_index = bus_data['index'][bus_data['Q_L'].values != 0].values
    # Q_L_values = bus_data['Q_L'][bus_data['Q_L'].values != 0].values

    Load_96_nor=Load_96.copy()/Load_96[0]  #归一化
    P_L_values_arr=np.zeros((num_rPoint,P_L_values.shape[0]))
    # P_G_values_arr = np.zeros((num_rPoint, P_G_values.shape[0]))
    # Q_L_values_arr = np.zeros((num_rPoint, P_L_values.shape[0]))

    #生成96个运行时刻的负荷功率
    for i in range(num_rPoint):
        P_L_valuesi=P_L_values*Load_96_nor[i]
        P_L_values_arr[i,]=P_L_valuesi
        # P_G_valuesi = P_G_values * Load_96_nor[i]
        # P_G_values_arr[i,] = P_G_valuesi
        # Q_L_valuesi = Q_L_values * Load_96_nor[i]
        # Q_L_values_arr[i,] = Q_L_valuesi

    #计算96个运行时刻的A矩阵及特征根
    vals=1j*np.zeros((num_rPoint,num_level))
    U,V=1j*np.zeros((num_rPoint,num_level,num_level)),1j*np.zeros((num_rPoint,num_level,num_level))
    bus_data_i = bus_data.copy()

    k=0
    for i in range(num_rPoint):
        bus_data_i["P_L"][P_L_index - 1] = P_L_values_arr[i,]
        # bus_data_i["P_G"][P_G_index - 1] = P_G_values_arr[i,]
        # bus_data_i["Q_L"][Q_L_index - 1] = Q_L_values_arr[i,]
        print(P_L_values_arr[i])
        A_mat_i=A_Generation(Power_flow,Y,bus_data_i,data_G,data_AVR,data_PM)
        vals[i,]=get_UV(A_mat_i)[0]
        U[i,],V[i,]=get_UV(A_mat_i)[1],get_UV(A_mat_i)[2]
        k+=1
        print(k)

    return vals,U,V


Eigenval=Eigenval_all(Load_96_ori,bus_data)
np.save("val",Eigenval[0])
np.save("U",Eigenval[1])
np.save("V",Eigenval[2])


#作图
r_vals_real=Eigenval[0].real.T.copy()
r_vals_imag=Eigenval[0].imag.T.copy()

plt.plot(r_vals_real,r_vals_imag,'o')
plt.ylabel('imag',verticalalignment='top')
plt.xlabel('real',horizontalalignment='right')
ax = plt.gca()                                            # get current axis 获得坐标轴对象

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')         # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')          # 指定下边的边作为 x 轴   指定左边的边为 y 轴

ax.spines['bottom'].set_position(('data', 0))   #指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
ax.spines['left'].set_position(('data', 0))

plt.show()