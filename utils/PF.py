#Power flow
import numpy as np
import pandas as pd
import copy
import time
np.set_printoptions(suppress=True)

from .YData import Y

start=time.time()
headers=['index','Node_type','U','delta','P_L','Q_L','P_G','Q_G','X','B']
Bus_data=pd.read_csv('data/Bus_data.csv',header=None)
Bus_data.columns = headers
bus_data=Bus_data.astype(float)

# print(bus_data)

num_Node=36

# 按节点类型分类，并保留原序号
index_bal = bus_data['index'][bus_data['Node_type'] == 3].values.astype(int)
index_PV = bus_data['index'][bus_data['Node_type'] == 2].values.astype(int)
index_PQ = np.sort(np.insert(bus_data['index'][bus_data['Node_type'] == 1].values, 0,
                             bus_data['index'][bus_data['Node_type'] == 0].values, 0).astype(int))

#将Y矩阵分块，将复数分解为2*2矩阵
def Y_seg(Y):
    shape_Y=Y.shape[0]
    Y_s = np.zeros((shape_Y*2,shape_Y*2))
    shape = (shape_Y, shape_Y, 2, 2)
    strides = Y_s.itemsize * np.array([shape_Y * 2, 2, shape_Y, 1])
    Y_s = np.lib.stride_tricks.as_strided(Y_s, shape=shape, strides=strides).copy()

    for i in range(shape_Y):
        for j in range(shape_Y):
            temp_mat=np.zeros((2,2))
            temp_mat[0][0] = Y[i][j].real
            temp_mat[1][1] = Y[i][j].real
            temp_mat[0][1] = -Y[i][j].imag
            temp_mat[1][0] = Y[i][j].imag
            Y_s[i,j] = temp_mat.copy()
    return Y_s

#将分块矩阵（4维）变为2*2矩阵(方阵)
def seg2mat(mat_seg):
    mat_s=mat_seg.copy()
    shape_out_row=mat_seg.shape[0]
    shape_out_col = mat_seg.shape[1]
    shape_in_row=mat_seg.shape[2]
    shape_in_col=mat_seg.shape[3]
    mat=np.zeros((shape_out_row*shape_in_row,shape_out_col*shape_in_col))
    for i in range(shape_out_row):
        for j in range(shape_in_row):
            mat[i*shape_in_row+j]=mat_s[i,:,j,:].flatten()
    return mat


#生成Ge-Bf 和Gf+Be 求和块
def Ge_sub_Bf_AND_Gf_plus_Be(Y,ef_mat):
    shape=Y.shape[0]
    #Y矩阵分块，变实数阵
    Y_s= Y_seg(Y)
    Y_real=seg2mat(Y_s)

    Ge_sub_Bf,Gf_plus_Be=[],[]
    for i in range(shape):
        Ge_sub_Bf_i = Y_real[i*2].dot(ef_mat)
        Gf_plus_Be_i = Y_real[i*2+1].dot(ef_mat)
        Ge_sub_Bf.append(Ge_sub_Bf_i)
        Gf_plus_Be.append(Gf_plus_Be_i)
    return Ge_sub_Bf,Gf_plus_Be

#生成雅可比矩阵,ef_mat为e1/f1/e2/f2顺序排列,为一维array,含平衡节点
def J_matrix(Y,ef_mat,Ge_sub_Bf,Gf_plus_Be):
    shape=Y.shape[0]
    #将J矩阵分块
    J=np.zeros((2*(num_Node-1),2*(num_Node-1)))
    shape_seg=(num_Node-1,num_Node-1,2,2)
    strides = J.itemsize * np.array([(num_Node-1)*2, 2, num_Node-1, 1])
    J_seg= np.lib.stride_tricks.as_strided(J, shape=shape_seg, strides=strides).copy()  # 不加copy，有bug

    #生成对角子块
    for i in range(shape):
        m=i
        if i not in index_bal-1:
            if i > (index_bal - 1)[0]:
                m -= 1
            Ge_sub_Bf_i =  Ge_sub_Bf[i]
            Gf_plus_Be_i = Gf_plus_Be[i]
            deltaP_deltae_i= -Ge_sub_Bf_i - Y[i][i].real*ef_mat[2*i] - Y[i][i].imag*ef_mat[2*i+1]
            deltaP_deltaf_i=  -Gf_plus_Be_i + Y[i][i].imag*ef_mat[2*i] - Y[i][i].real*ef_mat[2*i+1]
            deltaQ_deltae_i= Gf_plus_Be_i + Y[i][i].imag*ef_mat[2*i] - Y[i][i].real*ef_mat[2*i+1]
            deltaQ_deltaf_i= -Ge_sub_Bf_i + Y[i][i].real*ef_mat[2*i] + Y[i][i].imag*ef_mat[2*i+1]
            deltaV2_deltae_i= -2*ef_mat[2*i]
            deltaV2_deltaf_i = -2 * ef_mat[2 * i+1]
            if i in index_PQ-1:
                block_mat=np.array([[deltaP_deltae_i,deltaP_deltaf_i],[deltaQ_deltae_i,deltaQ_deltaf_i]])
                J_seg[m,m]=copy.deepcopy(block_mat)

            if i in index_PV-1:
                block_mat = np.array([[deltaP_deltae_i, deltaP_deltaf_i], [deltaV2_deltae_i, deltaV2_deltaf_i]])
                J_seg[m, m] = copy.deepcopy(block_mat)

    #生成非对角子块
    for i in range(shape):
        m = i
        if i not in index_bal - 1:
            if i > (index_bal - 1)[0]:
                m -= 1
            for j in range(shape):
                n = j
                if j not in index_bal - 1:
                    if j > (index_bal - 1)[0]:
                        n -= 1
                    if i!=j:
                        deltaP_deltae_i= - Y[i][j].real*ef_mat[2*i] - Y[i][j].imag*ef_mat[2*i+1]
                        deltaP_deltaf_i=  Y[i][j].imag*ef_mat[2*i] - Y[i][j].real*ef_mat[2*i+1]
                        deltaQ_deltae_i=  Y[i][j].imag*ef_mat[2*i] - Y[i][j].real*ef_mat[2*i+1]
                        deltaQ_deltaf_i= Y[i][j].real*ef_mat[2*i] + Y[i][j].imag*ef_mat[2*i+1]
                        deltaV2_deltae_i= 0
                        deltaV2_deltaf_i = 0
                        if i in index_PQ-1:
                            block_mat=np.array([[deltaP_deltae_i,deltaP_deltaf_i],[deltaQ_deltae_i,deltaQ_deltaf_i]])
                            J_seg[m,n]=copy.deepcopy(block_mat)
                        if i in index_PV-1:
                            block_mat = np.array([[deltaP_deltae_i, deltaP_deltaf_i], [deltaV2_deltae_i, deltaV2_deltaf_i]])
                            J_seg[m, n] = copy.deepcopy(block_mat)
    J=seg2mat(J_seg)
    return J

# print(J_matrix(Y,np.ones(num_Node*2)))

def Power_flow(Y,bus_data):
    #初始化电压值
    e,f = np.zeros(num_Node),np.zeros(num_Node)
    e[index_bal - 1] = bus_data['U'][index_bal - 1].values* np.cos(bus_data['delta'][index_bal - 1].values)
    e[index_PV - 1] = bus_data['U'][index_PV - 1].values
    e[index_PQ - 1] = 1
    f[index_bal - 1] = bus_data['U'][index_bal - 1].values* np.sin(bus_data['delta'][index_bal - 1].values)
    f[index_PV - 1] = 0
    f[index_PQ - 1] = 0
    ef_mat=np.zeros(num_Node*2)
    ef_mat[::2]=e.copy()
    ef_mat[1::2]=f.copy()

    P_s,Q_s,V2_s = np.zeros(num_Node),np.zeros(num_Node),np.zeros(num_Node)
    P_s[index_PV - 1] = bus_data['P_G'][index_PV - 1].values - bus_data['P_L'][index_PV - 1].values
    P_s[index_PQ - 1] = bus_data['P_G'][index_PQ - 1].values - bus_data['P_L'][index_PQ - 1].values

    Q_s[index_PQ - 1] = bus_data['Q_G'][index_PQ - 1].values - bus_data['Q_L'][index_PQ - 1].values

    V2_s[index_PV - 1] = bus_data['U'][index_PV - 1].values**2

    #循环
    k = 0  #k为循环次数
    deltaV = np.ones((num_Node * 2 - 2, 1))
    while np.max(abs(deltaV)) > pow(10, -8):
        Ge_sub_Bf,Gf_plus_Be=Ge_sub_Bf_AND_Gf_plus_Be(Y, ef_mat)
        deltaP=P_s-e*Ge_sub_Bf-f*Gf_plus_Be
        deltaQ=Q_s-f*Ge_sub_Bf+e*Gf_plus_Be
        deltaV2=V2_s - (e**2+f**2)
        deltaW=np.zeros((num_Node*2-2,1))
        for i in range(num_Node):
            m = i
            if i not in index_bal - 1:
                if i > (index_bal - 1)[0]:
                    m -= 1
                if i in index_PV-1:
                    deltaW[m*2]=deltaP[i]
                    deltaW[m*2+1]=deltaV2[i]
                if i in index_PQ-1:
                    deltaW[m*2]=deltaP[i]
                    deltaW[m*2+1]=deltaQ[i]

        J=J_matrix(Y,ef_mat,Ge_sub_Bf,Gf_plus_Be)

        deltaV=-np.linalg.inv(J).dot(deltaW)

        deltaU=np.zeros((num_Node*2,1))
        deltaU[2:]=deltaV
        e = e + deltaU[::2,0]
        f = f + deltaU[1::2,0]
        ef_mat = np.zeros(num_Node * 2)
        ef_mat[::2] = e.copy()
        ef_mat[1::2] = f.copy()
        # if k == 0:
            # print("deltaV",deltaV,e,f)
        k+=1
        # print("here",k)
    U=(e**2+f**2)**0.5
    delta=np.arctan(f/e)
    return e,f,U,delta

#print(Power_flow(Y,bus_data))
# end=time.time()
# print(end-start)