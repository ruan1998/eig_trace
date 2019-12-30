import pandas as pd
import numpy as np
import sympy as sp
from math import pi
from .YData import Y
from .PF import Power_flow
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
# data_PM['伺服机构时间常数T_S']=data_PM['伺服机构时间常数T_S'].values*100*pi
# data_PM['中间过热时间常数T_RH']=data_PM['中间过热时间常数T_RH'].values*100*pi
# data_PM['蒸汽容积时间常数T0_ST']=data_PM['蒸汽容积时间常数T0_ST'].values*100*pi
data_PM['标幺基值转换系数K_mH']=1


headers_AVR = ['index','type','T_A','T_E','T_F','K_A','K_F','T1','T2','T3','T4']

rows_AVR = [
        [1]+[np.nan]*5,
        [2,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [3,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [4,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [5,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [6,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [7,1,np.nan,0.02,np.nan,20,np.nan,2,2,2,2],
        [8,1,np.nan,0.02,np.nan,20,np.nan,2,2,2,2],  #7，8励磁机是1型（自并励）
]

# rows_AVR = [
#         [1]+[np.nan]*5,
#         [2,1,np.nan,0.02,np.nan,20,np.nan,2,2,2,2],
#         [3,1,np.nan,0.02,np.nan,20,np.nan,2,2,2,2],
#         [4,1,np.nan,0.02,np.nan,20,np.nan,2,2,2,2],
#         [5,1,np.nan,0.02,np.nan,20,np.nan,2,2,2,2],
#         [6,1,np.nan,0.02,np.nan,20,np.nan,2,2,2,2],
#         [7,1,np.nan,0.02,np.nan,20,np.nan,2,2,2,2],
#         [8,1,np.nan,0.02,np.nan,20,np.nan,2,2,2,2],  #7，8励磁机是1型（自并励）
# ]

rows_AVR = [
        [1]+[np.nan]*5,
        [2,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [3,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [4,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [5,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [6,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [7,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,
        [8,2,0.03,0.5,0.715,50,0.04]+[np.nan]*4,  #7，8励磁机是1型（自并励）
]

data_AVR=pd.DataFrame(columns=headers_AVR ,data=rows_AVR)
# data_AVR['T_A']=data_AVR['T_A'].values*100*pi
# data_AVR['T_E']=data_AVR['T_E'].values*100*pi
# data_AVR['T_F']=data_AVR['T_F'].values*100*pi


headers_G = ['index','X_d','X_d_p','X_d_pp','X_q','X_q_p','X_q_pp','T_d0_p','T_d0_pp','T_q0_p','T_q0_pp','M','D']

rows_G = [
        [1, 0.282,0.282,0.282,  0.282,0.282,0.282,  10,0.1,9999,0.2,        7.49,0],
        [2, 2.266,0.27,0.168,   2.266,2.266,0.168,  8.375,0.224,9999,1.66,  4.249,0],
        [3, 1.217,0.349,0.25,   0.6,0.6,0.25,       7.24,0.1,9999,0.2,      9.014,0],
        [4, 1.81,0.284,0.183,   1.81,1.81,0.183,    6.2,0.192,9999,1.89,    6.672,0],
        [5, 1.951,0.306,0.198,  1.951,1.951,0.198,  6.2,0.1,9999,0.5,       6.149,0],
        [6, 1.633,0.197,0.148,  1.633,1.633,0.148,  6.92,0.1,9999,0.2,      2.62,0],
        [7, 0.904,0.358,0.252,  0.64,0.64,0.252,    5.53,0.05,9999,0.05,    7.692,0],
        [8, 0.75,0.306,0.196,   0.611,0.611,0.196,  5.95,0.05,9999,0.05,    8.393,0],
]


data_G=pd.DataFrame(columns=headers_G ,data=rows_G)
# data_G['T_d0_p']=data_G['T_d0_p'].values*100*pi
# data_G['T_d0_pp']=data_G['T_d0_pp'].values*100*pi
# data_G['T_q0_p']=data_G['T_q0_p'].values*100*pi
# data_G['T_q0_pp']=data_G['T_q0_pp'].values
# data_G['M']=data_G['M'].values*100*pi


headers_PSS = ['index','K_P','K_w','K_PSS','T_w','T_1','T_2','T_3','T_4']

rows_PSS = [
        [1]+[0]*8,
        [2]+[0]*8,
        [3]+[0]*8,
        [4]+[0]*8,
        [5]+[0]*8,
        [6]+[0]*8,
        [7]+[0]*8,
        [8]+[0]*8,
]

# rows_PSS = [
#         [1]+[0]*8,
#         [2]+[0]*8,
#         [3]+[0]*8,
#         [4]+[0]*8,
#         [5]+[0]*8,
#         [6]+[0]*8,
#         [7]+[0]*8,
#         [8]+[0]*8,
# ]

data_PSS=pd.DataFrame(columns=headers_PSS ,data=rows_PSS)




headers=['index','Node_type','U','delta','P_L','Q_L','P_G','Q_G','X','B']
Bus_data=pd.read_csv("data/Bus_data.csv",header=None)
Bus_data.columns = headers
bus_data=Bus_data.astype(float)

# #f:f(x)=0
# def New_Raph(f,x,init_x):
#         f_diff_temp = sp.diff(f, x)
#         y=f.evalf(subs ={x:init_x})
#         f_diff_value=f_diff_temp.evalf(subs ={x:init_x})
#         deltax=y/f_diff_value
#         x_value=init_x-deltax
#         while deltax>0.1**8:
#                 y = f.evalf(subs={x: x_value})
#                 f_diff_value = f_diff_temp.evalf(subs={x: x_value})
#                 deltax = y / f_diff_value
#                 x_value -= deltax
#         return x_value
#
#
# def Xq_qq2Xq(ux,uy,Xq_pp,ix,iy,P,Q,init_val):
#         xq=sp.Symbol("xq")
#         delta=sp.Symbol("\delta")
#         delta_subs=sp.atan(P*xq/(ux**2+uy**2+Q*xq))
#         f=((sp.sin(delta)*ux-sp.cos(delta)*uy)/(xq-Xq_pp)-(2*Xq_pp+xq)/(xq+Xq_pp)*(sp.cos(delta)*ix+sp.sin(delta)*iy)).subs({delta:delta_subs})
#         # x_val = New_Raph(f, xq, init_val)
#
#         x_val=np.array([i for i in range(2500)])*0.002
#         y=np.zeros(2500)
#         for i in range(2500):
#                 y[i]=f.evalf(subs={xq:x_val[i]})
#                 if(y[i]*y[i-1]<0):
#                         print(x_val[i])
#         plt.plot(x_val,y)
#         plt.show()
#         return f,x_val,xq
#
# Ux,Uy,Ut,xita=Power_flow(Y,bus_data)
# U_mat=Ux+1j*Uy
# I_mat=Y.dot(U_mat)  #求得注入电流
# ux=Ux[:8].real
# uy=Ux[:8].imag
# Xq_pp=np.array([0.282,0.168,0.25,0.183,0.198,0.148,0.252,0.196])
# ix=I_mat[:8].real
# iy=I_mat[:8].imag
# P=np.array([6.14933,6,3.1,1.6,4.3,-0.01,2.25,3.06])
# Q=np.array([1.95138,3.6,2.5691,0.7,3.34,0.70155,0.32119,0.4354])
#
# # f,x_val,xq=Xq_qq2Xq(ux[4],uy[4],Xq_pp[4],ix[4],iy[4],P[4],Q[4],0)
#
# # print(x_val)
# # print(f.evalf(subs={xq:x_val}))
#
# # Xq=[0.282,1.022,0.168,0.25,0.184,0.198]


