import pandas as pd
import numpy as np
import sympy as sp
import math
from .YData import Y
from PF import Power_flow
import matplotlib.pyplot as plt

headers_PM = ['index','调差系数','标幺基值转换系数K_mH','伺服机构时间常数T_S','汽轮机过热系数alpha','中间过热时间常数T_RH',
           '蒸汽容积时间常数T0_ST','T_W_ST = 0','K_i_ST = 1',
           '水锤效应时间常数T_W_WT','水轮机软反馈T_i_WT','软反馈放大倍数K_beta','量测环节放大倍数K_delta_WT']

rows_PM = [
        [1,np.nan,18.8]+[np.nan]*10,
        [2,0.05,7.8,0.03,0.3,5,0.5,0,1]+[np.nan]*4,
        [3,0.04,8.82,0.03,0.3,5,0.5,0,1]+[np.nan]*4,
        [4,0.05,2.35,0.03,0.3,5,0.5,0,1]+[np.nan]*4,
        [5,0.05,6.375,0.03,0.3,5,0.5,0,1]+[np.nan]*4,
        [6,np.nan,3.5]+[np.nan]*10,
        [7,0.04,2.86,0.03,0.3,5,0.5,0,1]+[np.nan]*4,
        [8,0.04,3.884,0.03,0.3,5,0.5,0,1]+[np.nan]*4
    ]

data_PM=pd.DataFrame(columns=headers_PM,data=rows_PM)

headers_AVR = ['index','T_A','T_E','T_F','K_A','K_F']

rows_AVR = [
        [1]+[np.nan]*5,
        [2,0.03,0.5,0.715,50,0.04],
        [3,0.03,0.5,0.715,50,0.04],
        [4,0.03,0.5,0.715,50,0.04],
        [5,0.03,0.5,0.715,50,0.04],
        [6,0.03,0.5,0.715,50,0.04],
        [7,0.03,0.5,0.715,50,0.04],
        [8,0.03,0.5,0.715,50,0.04],  #先假设7，8励磁机是I型，且参数与前述一致
]

data_AVR=pd.DataFrame(columns=headers_AVR ,data=rows_AVR)

headers_G = ['index','X_d','X_d_sk','X_q','M','D','T_d0']  #M,T_d0是s为单位，不是标幺值(要*100π)

rows_G = [
        [1,0.282,0.282,0.282,7.49,0,10],
        [2,2.266,0.27,2.266,4.249,0,8.375 ],
        [3,1.217,0.349,0.6,9.014 ,0,7.24 ],
        [4,1.81,0.284,1.81,6.672 ,0,6.2 ],
        [5,1.951,0.306,1.951,6.149 ,0,6.2 ],
        [6,1.633,0.197,1.633,2.62 ,0,6.92 ],
        [7,0.904,0.358,0.64,7.692 ,0,5.53 ],
        [8,0.75,0.306,0.611,8.393 ,0,5.95 ],
]

data_G=pd.DataFrame(columns=headers_G ,data=rows_G)
# data_G['T_d0']=data_G['T_d0'].values*314
# data_G['M']=data_G['M'].values*314

headers=['index','Node_type','U','delta','P_L','Q_L','P_G','Q_G','X','B']
Bus_data=pd.read_csv('data/Bus_data.csv',header=None)
Bus_data.columns = headers
bus_data=Bus_data.astype(float)

def gaussian_elim(M, n):
    """
    高斯消去负荷节点
    parameters
    ==========
    M 待高斯消元的方阵
    n 保留的左上角n * n发电机节点方阵
    """
    M = M.copy().astype(np.float32)
    m = M.shape[0] - n
    for i in range(1, m+1):
        if M[-i, -i] != 0:
            reduce_array = M[-i] * (M[:-i, -i] / M[-i, -i])[:, None]
            M[:-i] = M[:-i] - reduce_array
        else: #交换行
            for j in range(1, m-i):
                if M[-j-1, -i] != 0:
                    M[-i, :], M[-j-1, :] = M[-j-1, :].copy(), M[-i, :].copy()
                    reduce_array = M[-i] * (M[:-i, -i] / M[-i, -i])[:, None]
                    M[:-i] = M[:-i] - reduce_array
                    break
    return M[:n,:n]
#
# M=np.array([[1,1,1,1],[2,2,2,4],[4,4,2,1],[1,2,1,2]])
# print(gaussian_elim(M,2))


