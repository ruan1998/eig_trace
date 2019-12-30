import sympy as sp
import numpy as np
import pandas as pd
from math import pi
from .YData import Y
from .DATA_DM2 import data_PM,data_G,data_AVR,data_PSS
from .PF import Power_flow,Y_seg,seg2mat
import matplotlib.pyplot as plt
import csv

np.set_printoptions(precision=6, suppress=True)

headers=['index','Node_type','U','delta','P_L','Q_L','P_G','Q_G','X','B']
Bus_data=pd.read_csv('data/Bus_data.csv',header=None)
Bus_data.columns = headers
bus_data=Bus_data.astype(float)


num_Node=bus_data.shape[0]
num_G=8

# 按节点类型分类，并保留原序号
index_bal = bus_data['index'][bus_data['Node_type'] == 3].values.astype(int)
index_PV = bus_data['index'][bus_data['Node_type'] == 2].values.astype(int)
index_PQ = np.sort(np.insert(bus_data['index'][bus_data['Node_type'] == 1].values, 0,
                             bus_data['index'][bus_data['Node_type'] == 0].values, 0).astype(int))

def get_UV(A):
    """
    获得A矩阵左右特征向量
    *** 特征向量矩阵中特征向量均为列向量 ***
    """
    vals, U = np.linalg.eig(A)
    V=np.linalg.inv(U).T.copy()
    return vals,U,V


#使矩阵分块
def seg_matrix(M, rows, cols):
    M=M.copy()
    r, c = M.shape
    reshape = int(r / rows), int(c / cols), rows, cols
    cut_pos = c * rows, cols, c, 1
    strides = M.itemsize * np.array(cut_pos)
    seg = np.lib.stride_tricks.as_strided(M, shape=reshape, strides=strides)
    return seg

#填充分块空矩阵
def subs_zeros_array(out_size=3, in_size=8, mode=None, **kwargs):
    """
    parameters
    ==========
    out_size 分块方阵外层大小
    in_size 分块方阵中每块方阵大小
    mode 选择 None 、'diag' 、'diag_col'模式
    kwargs 指定替换位置的矩阵或者其它模式下的信息
    """
    size = out_size * in_size
    A_part = np.zeros((size, size))
    if mode is None:
        for pos, arr in kwargs.items():
            row, col = int(pos[1]), int(pos[2])
            A_part[in_size*row:in_size*(row+1), in_size*col:in_size*(col+1)] = arr
    elif mode == 'diag' or mode == 'diag_col':
        s = kwargs['start']
        array = kwargs['array']
        array_len = array.shape[0] if mode=='diag' else array.shape[-1]
        if  array_len > out_size - s:
            print("[Error] array's elements are too many")
            return None
        for i in range( array_len):
            start, end = in_size*(i+s), in_size*(i+s+1)
            A_part[start:end, start:end] = array[i] if mode=='diag' else array[...,i]
    return A_part

#高斯消去
def gaussian_elim(M, n):
    """
    高斯消去负荷节点
    parameters
    ==========
    M 待高斯消元的方阵
    n 保留的左上角n * n发电机节点方阵
    """
    M1 = M.copy().astype(np.float32)
    m = M.shape[0] - n
    for i in range(1, m+1):
        if M1[-i, -i] != 0:
            reduce_array = M1[-i] * (M1[:-i, -i] / M1[-i, -i])[:, None]
            M1[:-i] = M1[:-i] - reduce_array
        else: #交换行
            for j in range(1, m-i+1):
                if M1[-i-j, -i] != 0:
                    temp=M1[-j-i, :].copy()
                    M1[-j-i, :] = M1[-i, :].copy()
                    M1[-i, :]=temp.copy()
                    reduce_array = M1[-i] * (M1[:-i, -i] / M1[-i, -i])[:, None]
                    M1[:-i] = M1[:-i] - reduce_array
                    break
                else:
                    if j==m-i:
                        print("failed")
                        return
    return M1[:n,:n]


#定义公式中的杂散参数的符号
#发电机定子电压方程中
#生成基本符号
def G_symbol():
    X_q_pp = sp.symbols('X_q_pp')
    X_q_p = sp.symbols('X_q_p')
    X_d_pp = sp.symbols('X_d_pp')  # 此处X_d_pp代表原公式的X_d''
    X_d_p = sp.symbols('X_d_p')    # 此处X_d_p代表原公式的X_d'

    delta = sp.symbols('/delta')  # 此处delta代表的是实际的delta值（不是△δ）
    E_q_pp = sp.symbols('E_q_pp')  # 此处E_q_pp代表原公式的E_q''
    E_q_p = sp.symbols('E_q_p')  # 此处E_q_p代表原公式的E_q'
    E_d_pp = sp.symbols('E_d_pp')
    E_d_p = sp.symbols('E_d_p')

    U_x = sp.symbols('U_x')
    U_y = sp.symbols('U_y')

    return X_q_pp,X_q_p, X_d_pp,X_d_p,delta,E_q_pp,E_q_p,E_d_pp,E_d_p,U_x,U_y

# 生成GB_F矩阵表达式(单台发电机)
def GB_F_matrix_sym():
    X_q_pp,X_d_pp,delta=G_symbol()[0:5:2]

    matrix_bulidGB1 = sp.Matrix(
        [[sp.sin(delta), sp.cos(delta)], [-sp.cos(delta), sp.sin(delta)]])  # 用sympy自带的Matrix函数创建矩阵
    matrix_bulidGB2 = sp.Matrix([[0, X_q_pp], [-X_d_pp, 0]])
    GB_matrix = 1 / (X_d_pp * X_q_pp) * matrix_bulidGB1 * matrix_bulidGB2 * matrix_bulidGB1.inv()  # sympy提供inv

    return GB_matrix

# 生成aq，bq，ad，bd,aδ，bδ的表达式
def ab_dq_And_delta_temp():
    X_q_pp, X_q_p, X_d_pp, X_d_p, delta, E_q_pp, E_q_p, E_d_pp, E_d_p, U_x, U_y=G_symbol()
    GB_F_matrix_temp = GB_F_matrix_sym()
    G_F1 = GB_F_matrix_temp[0, 0]
    G_F2 = GB_F_matrix_temp[1, 1]
    B_F1 = -GB_F_matrix_temp[0, 1]
    B_F2 = GB_F_matrix_temp[1, 0]
    G_F1_diff = sp.diff(G_F1, delta)  # 对delta求导
    G_F2_diff = sp.diff(G_F2, delta)
    B_F1_diff = sp.diff(B_F1, delta)
    B_F2_diff = sp.diff(B_F2, delta)

    a_q = G_F1 * sp.cos(delta) - B_F1 * sp.sin(delta)
    b_q = B_F2 * sp.cos(delta) + G_F2 * sp.sin(delta)

    a_d = G_F1 * sp.sin(delta) + B_F1 * sp.cos(delta)
    b_d = B_F2 * sp.sin(delta) - G_F2 * sp.cos(delta)

    a_delta = E_q_pp * ((G_F1_diff - B_F1) * sp.cos(delta)- (B_F1_diff + G_F1) * sp.sin(delta)) \
              +E_d_pp*((G_F1+B_F1_diff)*sp.cos(delta)+(G_F1_diff-B_F1)*sp.sin(delta))  - (G_F1_diff * U_x - B_F1_diff * U_y)
    b_delta = E_q_pp * ((B_F2 + G_F2) * sp.cos(delta)- (G_F2_diff - B_F2) * sp.sin(delta)) \
              +E_d_pp*((B_F2-G_F2_diff)*sp.cos(delta)+(B_F2_diff+G_F2)*sp.sin(delta)) - (B_F2_diff * U_x + G_F2_diff * U_y)

    return a_q,b_q,a_d,b_d,a_delta,b_delta

#得到符号
GB_F_sym=GB_F_matrix_sym()
ab_dq_And_delta_symbol=ab_dq_And_delta_temp()

#输出GB_F列表，按发电机序号的顺序排列好各个GB_F矩阵
def GB_F_matrix_list(delta_array, X_d_pp_array, X_q_pp_array):
    X_q_pp, X_q_p, X_d_pp, X_d_p, delta, E_q_pp, E_q_p, E_d_pp, E_d_p, U_x, U_y=G_symbol()
    GB_F_sym = GB_F_matrix_sym()

    # 生成aq，bq，adelta,bdelta的真实值
    GB_F_temp = sp.lambdify((delta, X_d_pp, X_q_pp), GB_F_sym, 'numpy')  # 用lambdify实现值替换符号

    GB_F_value = GB_F_temp(delta_array, X_d_pp_array, X_q_pp_array)

    GB_F=[]
    for i in range(num_G):
        GB_F.append(GB_F_value[...,i])
    return GB_F

def ab_dq_And_delta(delta_array, X_d_pp_array, X_q_pp_array, E_q_pp_array, E_d_pp_array,U_x_array, U_y_array):
    X_q_pp, X_q_p, X_d_pp, X_d_p, delta, E_q_pp, E_q_p, E_d_pp, E_d_p, U_x, U_y=G_symbol()
    a_q, b_q, a_d, b_d, a_delta, b_delta = ab_dq_And_delta_temp()
    # 生成aq，bq，ad,bd,adelta,bdelta的真实值
    a_q_temp = sp.lambdify((delta, X_d_pp, X_q_pp), a_q, 'numpy')  # 用lambdify实现值替换符号
    b_q_temp = sp.lambdify((delta, X_d_pp, X_q_pp), b_q, 'numpy')

    a_q_value = a_q_temp(delta_array, X_d_pp_array, X_q_pp_array)  # a_q_value是array
    b_q_value = b_q_temp(delta_array, X_d_pp_array, X_q_pp_array)

    a_d_temp = sp.lambdify((delta, X_d_pp, X_q_pp), a_d, 'numpy')  # 用lambdify实现值替换符号
    b_d_temp = sp.lambdify((delta, X_d_pp, X_q_pp), b_d, 'numpy')

    a_d_value = a_d_temp(delta_array, X_d_pp_array, X_q_pp_array)  # a_q_value是array
    b_d_value = b_d_temp(delta_array, X_d_pp_array, X_q_pp_array)

    a_delta_temp = sp.lambdify((delta, X_d_pp, X_q_pp, E_q_pp, E_d_pp, U_x, U_y), a_delta, 'numpy')  # 用lambdify实现值替换符号
    b_delta_temp = sp.lambdify((delta, X_d_pp, X_q_pp, E_q_pp, E_d_pp, U_x, U_y), b_delta, 'numpy')

    a_delta_value = a_delta_temp(delta_array, X_d_pp_array, X_q_pp_array, E_q_pp_array, E_d_pp_array, U_x_array, U_y_array)
    b_delta_value = b_delta_temp(delta_array, X_d_pp_array, X_q_pp_array, E_q_pp_array, E_d_pp_array, U_x_array, U_y_array)

    ab_q_single_line = np.array([a_q_value, b_q_value])
    ab_d_single_line = np.array([a_d_value, b_d_value])
    ab_delta_single_line = np.array([a_delta_value, b_delta_value])

    return ab_q_single_line, ab_d_single_line, ab_delta_single_line

#生成cq,dq,cd,dd, c_delta,d_delta矩阵
def cd_q_And_delta(delta_array, X_d_pp_array, X_q_pp_array, E_q_pp_array, E_d_pp_array, U_x_array, U_y_array,matrix_RX):
    # 从ab_q_And_delta得到
    ab_q_single_line, ab_d_single_line,ab_delta_single_line = ab_dq_And_delta(delta_array, X_d_pp_array,
                                                                 X_q_pp_array, E_q_pp_array, E_d_pp_array, U_x_array, U_y_array)

    #根据式12-25a，生成cq，dq，cδ，dδ
    #对矩阵分块
    ab_q_matrix=seg_matrix(ab_q_single_line,2,1)
    ab_d_matrix = seg_matrix(ab_d_single_line, 2, 1)
    ab_delta_matrix=seg_matrix(ab_delta_single_line,2,1)
    matrix_RX_seg=seg_matrix(matrix_RX,2,2)

    cd_q_matrix_seg=np.matmul(matrix_RX_seg,ab_q_matrix)  #利用matmul的特点
    cd_d_matrix_seg = np.matmul(matrix_RX_seg, ab_d_matrix)
    cd_delta_matrix_seg=np.matmul(matrix_RX_seg, ab_delta_matrix)
    c_q_matrix =cd_q_matrix_seg[..., 0, 0]
    d_q_matrix =cd_q_matrix_seg[...,1,0]
    c_d_matrix = cd_d_matrix_seg[..., 0, 0]
    d_d_matrix = cd_d_matrix_seg[..., 1, 0]
    c_delta_matrix = cd_delta_matrix_seg[..., 0, 0]
    d_delta_matrix = cd_delta_matrix_seg[..., 1, 0]

    #输出前三项为cd共同矩阵（已分块）。后六项分别是六个分立矩阵
    return cd_q_matrix_seg,cd_d_matrix_seg,cd_delta_matrix_seg,\
           c_q_matrix,d_q_matrix,c_d_matrix,d_d_matrix,c_delta_matrix,d_delta_matrix

#生成eq,fq,e_delta,f_delta矩阵
def ef_q_And_delta(delta_array, X_d_pp_array, X_q_pp_array, E_q_pp_array, E_d_pp_array, U_x_array, U_y_array,matrix_RX):
    cd_q_matrix,cd_d_matrix,cd_delta_matrix=cd_q_And_delta(delta_array, X_d_pp_array, X_q_pp_array,
                                               E_q_pp_array, E_d_pp_array, U_x_array, U_y_array,matrix_RX)[0:3]
    GB_F_mat_list=GB_F_matrix_list(delta_array,X_d_pp_array,X_q_pp_array)

    #为简化式12-26a,构造一个大GB_F_matrix_seg矩阵，用分块矩阵乘法(可再加速)
    GB_F_matrix = np.zeros((num_G * 2, num_G* 2))
    GB_F_matrix_seg = seg_matrix(GB_F_matrix,2,2)
    for i in range(num_G):
        GB_F_matrix_seg[i,:] = GB_F_mat_list[i]

    ef_q_matrix_seg=np.matmul(-GB_F_matrix_seg,cd_q_matrix)
    ef_d_matrix_seg = np.matmul(-GB_F_matrix_seg, cd_d_matrix)
    ef_delta_matrix_seg = np.matmul(-GB_F_matrix_seg, cd_delta_matrix)

    e_q_matrix =ef_q_matrix_seg[..., 0, 0]
    f_q_matrix =ef_q_matrix_seg[...,1,0]
    e_d_matrix = ef_d_matrix_seg[..., 0, 0]
    f_d_matrix = ef_d_matrix_seg[..., 1, 0]
    e_delta_matrix = ef_delta_matrix_seg[..., 0, 0]
    f_delta_matrix = ef_delta_matrix_seg[..., 1, 0]

    ab_q,ab_d,ab_delta=ab_dq_And_delta(delta_array, X_d_pp_array, X_q_pp_array, E_q_pp_array, E_d_pp_array, U_x_array, U_y_array)
    a_q,b_q=ab_q[0],ab_q[1]
    a_d, b_d = ab_d[0], ab_d[1]
    a_delta, b_delta = ab_delta[0], ab_delta[1]
    e_q_matrix+=np.diag(a_q)
    f_q_matrix += np.diag(b_q)
    e_d_matrix+=np.diag(a_d)
    f_d_matrix += np.diag(b_d)
    e_delta_matrix += np.diag(a_delta)
    f_delta_matrix += np.diag(b_delta)

    return e_q_matrix,f_q_matrix,e_d_matrix,f_d_matrix,e_delta_matrix,f_delta_matrix

def xy2dq(Ax,Ay,delta):
    Ad,Aq=np.zeros(delta.shape[0]),np.zeros(delta.shape[0])
    for i in range(delta.shape[0]):
        trans_mat=np.array([[np.sin(delta[i]),-np.cos(delta[i])],[np.cos(delta[i]),np.sin(delta[i])]])
        A=np.array([[Ax[i]],[Ay[i]]])
        Ad[i],Aq[i]= trans_mat.dot(A)[0,0],trans_mat.dot(A)[1,0]
    return Ad,Aq

def A_Generation(Power_flow,Y,bus_data,data_G,data_AVR,data_PM):
    # 发电机参数
    X_d_array = data_G['X_d'].values
    X_d_p_array = data_G['X_d_p'].values
    X_d_pp_array = data_G['X_d_pp'].values
    X_q_array = data_G['X_q'].values
    X_q_p_array = data_G['X_q_p'].values
    X_q_pp_array = data_G['X_q_pp'].values
    M_array = data_G['M'].values
    D_array = data_G['D'].values
    T_d0_p_array = data_G['T_d0_p'].values
    T_d0_pp_array = data_G['T_d0_pp'].values
    T_q0_p_array = data_G['T_q0_p'].values
    T_q0_pp_array = data_G['T_q0_pp'].values

    Ux, Uy, Ut, xita = Power_flow(Y, bus_data)

    U_mat = Ux + 1j * Uy
    I_mat = Y.dot(U_mat)  # 求得注入电流

    S=U_mat*(I_mat.real-1j*I_mat.imag)

    K_mH=data_PM['标幺基值转换系数K_mH'].values
    P_G=S.real[:num_G]/K_mH
    Q_G=S.imag[:num_G]/K_mH

    U_x_array=Ux[:num_G]
    U_y_array=Uy[:num_G]
    I_x_array=I_mat.real[:num_G]/K_mH
    I_y_array=I_mat.imag[:num_G]/K_mH
    delta_array=np.arctan(P_G*X_q_array/(Ut[:num_G]**2+Q_G*X_q_array)) + xita[:num_G]  #rad单位,要加上U角度，以平衡机为基准


    U_d_array,U_q_array=xy2dq(U_x_array,U_y_array,delta_array)
    I_d_array,I_q_array=xy2dq(I_x_array,I_y_array,delta_array)

    # I_d_array = I_d_array.copy()
    # I_q_array = I_q_array.copy() / np.array([18.8, 7.8, 8.82, 2.35, 6.375, 3.5, 2.86, 3.884])


    E_q_pp_array = U_q_array + I_d_array * X_d_pp_array
    E_d_pp_array = U_d_array - I_q_array * X_q_pp_array
    # print(delta_array,E_q_pp_array,E_d_pp_array)

    #here
    # delta_array=np.array([0.09119,0.42817,-0.29184,0.30112,0.0117,-0.66506,0.23189,0.21817])
    # E_q_pp_array=np.array([1.018,0.98538,1.1313,0.92909,1.0581,1.0723,1.0582,1.0415])
    # E_d_pp_array=np.array([0,0.581,0.09734,0.58772,0.48373,-0.00327,0.25341,0.27385])

    # 励磁参数
    T_A_array = data_AVR['T_A'].values
    K_A_array = data_AVR['K_A'].values
    T_E_array = data_AVR['T_E'].values
    K_F_array = data_AVR['K_F'].values
    T_F_array = data_AVR['T_F'].values
    T1_array = data_AVR['T1'].values
    T2_array = data_AVR['T2'].values
    T3_array = data_AVR['T3'].values
    T4_array = data_AVR['T4'].values

    def change_2_STorWT(A_arr,B_arr):  #生成一个矩阵，使得B_arr中不为nan的位置由A_arr对应元素填充
        A_arr_changed = np.ones(B_arr.shape[0]) * np.nan
        A_arr_changed[~np.isnan(B_arr)] = A_arr[~np.isnan(B_arr)]
        return A_arr_changed

    #原动机、调速器参数
    adjust_coef_array=data_PM['调差系数'].values #调差系数
    K_mH_array=data_PM['标幺基值转换系数K_mH'].values #标幺基值转换系数，SR/SB,机组额定容量/系统容量基值
    T_S_array=data_PM['伺服机构时间常数T_S'].values

    #汽轮机
    alpha_array=data_PM['汽轮机过热系数alpha'].values
    T_RH_array=data_PM['中间过热时间常数T_RH'].values
    T0_array_ST=data_PM['蒸汽容积时间常数T0_ST'].values
    T_W_array_ST=data_PM['T_W_ST = 0'].values  #汽轮机，T_W_ST = 0
    adjust_coef_array_ST=change_2_STorWT(adjust_coef_array,alpha_array)
    K_delta_array_ST=1/adjust_coef_array_ST #由调差系数算得，调差系数=1/K_delta
    K_i_array_ST=data_PM['K_i_ST = 1'].values #硬反馈放大倍数，1
    K_mH_array_ST=change_2_STorWT(K_mH_array,alpha_array)
    T_S_array_ST=change_2_STorWT(T_S_array,alpha_array)

    #水轮机
    T_W_array_WT=data_PM['水锤效应时间常数T_W_WT'].values
    T0_array_WT=T_W_array_WT*0.5  #T0=0.5*T_W
    T_i_array=data_PM['水轮机软反馈T_i_WT'].values
    adjust_coef_array_WT=change_2_STorWT(adjust_coef_array,T_W_array_WT)
    K_beta_array=data_PM['软反馈放大倍数K_beta'].values   #软反馈放大倍数Kb
    K_delta_array_WT=data_PM['量测环节放大倍数K_delta_WT'].values  #量测环节放大倍数，Ka
    K_i_array_WT= K_delta_array_WT*adjust_coef_array_WT  #由调差系数算得，调差系数=Ki/K_delta
    K_mH_array_WT=change_2_STorWT(K_mH_array,T_W_array_WT)
    T_S_array_WT=change_2_STorWT(T_S_array,T_W_array_WT)

    # PSS参数
    index_PSS=data_PSS['index'].values
    K_PSS_array=data_PSS['K_PSS'].values
    K_w_array=data_PSS['K_w'].values
    K_P_array=data_PSS['K_P'].values
    T_w_array = data_PSS['T_w'].values
    T1_PSS_array = data_PSS['T_1'].values
    T2_PSS_array = data_PSS['T_2'].values
    T3_PSS_array = data_PSS['T_3'].values
    T4_PSS_array = data_PSS['T_4'].values
    a_array=np.zeros(index_PSS.shape)
    for i in range(index_PSS.shape[0]):
        if T2_PSS_array[i]!=0:
            a_array[i]=T1_PSS_array[i]/T2_PSS_array[i]
        else:
            a_array[i]=0


    # RX_martix生成,P154
    PL_array = bus_data['P_L'].values[num_G:]
    QL_array = bus_data['Q_L'].values[num_G:]

    PL, QL = sp.symbols('P_L,Q_L')
    Ux_sy, Uy_sy = sp.symbols('U_sx, U_sy')

    GB = sp.Matrix([[PL / (Ux_sy ** 2 + Uy_sy ** 2), QL / (Ux_sy ** 2 + Uy_sy ** 2)],
                    [-QL / (Ux_sy ** 2 + Uy_sy ** 2), PL / (Ux_sy ** 2 + Uy_sy ** 2)]])
    GB_template = sp.lambdify((Ux_sy, Uy_sy, PL, QL), GB, 'numpy')

    GB_value = GB_template(Ux[num_G:], Uy[num_G:], PL_array, QL_array)

    GBL_mat = subs_zeros_array(num_Node, 2, mode="diag_col", array=GB_value, start=num_G)

    Y_s = Y_seg(Y).copy()
    Y_ConNode = seg2mat(Y_s) + GBL_mat
    Y_star = gaussian_elim(Y_ConNode, num_G * 2)

    GB_F_arr = np.zeros((2, 2, num_G))
    for i in range(2):
        for j in range(2):
            for m in range(num_G):
                GB_F_arr[i, j, m] = GB_F_matrix_list(delta_array, X_d_pp_array, X_q_pp_array)[m][i, j]
    GB_F_mat = subs_zeros_array(num_G, 2, mode="diag_col", array=GB_F_arr, start=0)
    Y_delta = Y_star + GB_F_mat
    RX_martix = np.linalg.inv(Y_delta)

    c_q_matrix, d_q_matrix, c_d_matrix, d_d_matrix,c_delta_matrix, d_delta_matrix = \
        cd_q_And_delta(delta_array, X_d_pp_array, X_q_pp_array,E_q_pp_array, E_d_pp_array, U_x_array, U_y_array, RX_martix)[3:]
    e_q_matrix, f_q_matrix, e_d_matrix, f_d_matrix,e_delta_matrix, f_delta_matrix = \
        ef_q_And_delta(delta_array, X_d_pp_array, X_q_pp_array,E_q_pp_array, E_d_pp_array, U_x_array, U_y_array, RX_martix)


    def U_I_G_DigMat():
        U_x_G_DigMat = np.diag(U_x_array)
        U_y_G_DigMat = np.diag(U_y_array)
        I_x_G_DigMat = np.diag(I_x_array)
        I_y_G_DigMat = np.diag(I_y_array)
        return U_x_G_DigMat, U_y_G_DigMat, I_x_G_DigMat, I_y_G_DigMat

    def K1_matrix():
        U_x_GDigMat,U_y_GDigMat,I_x_GDigMat,I_y_GDigMat=U_I_G_DigMat()
        K1=np.dot(U_x_GDigMat,e_delta_matrix)+np.dot(U_y_GDigMat,f_delta_matrix)\
           +np.dot(I_x_GDigMat,c_delta_matrix)+np.dot(I_y_GDigMat,d_delta_matrix)
        return K1


    def K2_matrix():
        U_x_GDigMat,U_y_GDigMat,I_x_GDigMat,I_y_GDigMat=U_I_G_DigMat()
        K2=np.dot(U_x_GDigMat,e_q_matrix)+np.dot(U_y_GDigMat,f_q_matrix)\
           +np.dot(I_x_GDigMat,c_q_matrix)+np.dot(I_y_GDigMat,d_q_matrix)
        return K2

    def K3_matrix():
        U_x_GDigMat,U_y_GDigMat,I_x_GDigMat,I_y_GDigMat=U_I_G_DigMat()
        K3=np.dot(U_x_GDigMat,e_d_matrix)+np.dot(U_y_GDigMat,f_d_matrix)\
           +np.dot(I_x_GDigMat,c_d_matrix)+np.dot(I_y_GDigMat,d_d_matrix)
        return K3

    def K4_matrix():
        U_t_array=(U_x_array**2+U_y_array**2)**0.5
        K4=np.dot(np.diag(U_x_array/U_t_array),c_delta_matrix)+np.dot(np.diag(U_y_array/U_t_array),d_delta_matrix)
        return K4

    def K5_matrix():
        U_t_array=(U_x_array**2+U_y_array**2)**0.5
        K5=np.dot(np.diag(U_x_array/U_t_array),c_q_matrix)+np.dot(np.diag(U_y_array/U_t_array),d_q_matrix)
        return K5

    def K6_matrix():
        U_t_array=(U_x_array**2+U_y_array**2)**0.5
        K6=np.dot(np.diag(U_x_array/U_t_array),c_d_matrix)+np.dot(np.diag(U_y_array/U_t_array),d_d_matrix)
        return K6

    def X1_to_3():
        T_d0_pp_inv = np.diag(1 / T_d0_pp_array)
        X1_matrix=-np.diag(X_d_p_array-X_d_pp_array).dot(T_d0_pp_inv).\
            dot((np.diag(np.sin(delta_array)).dot(e_delta_matrix))-(np.diag(np.cos(delta_array)).dot(f_delta_matrix))
                +np.diag(np.cos(delta_array)*I_x_array+np.sin(delta_array)*I_y_array))

        X2_matrix=-np.diag(X_d_p_array-X_d_pp_array).dot(T_d0_pp_inv).\
            dot((np.diag(np.sin(delta_array)).dot(e_q_matrix))-(np.diag(np.cos(delta_array)).dot(f_q_matrix)))- T_d0_pp_inv

        X3_matrix=-np.diag(X_d_p_array-X_d_pp_array).dot(T_d0_pp_inv).\
            dot((np.diag(np.sin(delta_array)).dot(e_d_matrix))-(np.diag(np.cos(delta_array)).dot(f_d_matrix)))

        return X1_matrix,X2_matrix,X3_matrix

    def X4_to_6():
        T_q0_pp_inv = np.diag(1 / T_q0_pp_array)
        X4_matrix=np.diag(X_q_p_array-X_q_pp_array).dot(T_q0_pp_inv).\
            dot((np.diag(np.cos(delta_array)).dot(e_delta_matrix))+(np.diag(np.sin(delta_array)).dot(f_delta_matrix))
                +np.diag(-np.sin(delta_array)*I_x_array+np.cos(delta_array)*I_y_array))

        X5_matrix=np.diag(X_q_p_array-X_q_pp_array).dot(T_q0_pp_inv).\
            dot((np.diag(np.cos(delta_array)).dot(e_q_matrix))+(np.diag(np.sin(delta_array)).dot(f_q_matrix)))

        X6_matrix=np.diag(X_q_p_array-X_q_pp_array).dot(T_q0_pp_inv).\
            dot((np.diag(np.cos(delta_array)).dot(e_d_matrix))+(np.diag(np.sin(delta_array)).dot(f_d_matrix))) - T_q0_pp_inv

        return X4_matrix,X5_matrix,X6_matrix

    def X7_to_9():
        T_d0_p_inv = np.diag(1 / T_d0_p_array)
        X7_matrix=-np.diag(X_d_array - X_d_p_array).dot(T_d0_p_inv). \
            dot((np.diag(np.sin(delta_array)).dot(e_delta_matrix)) - (np.diag(np.cos(delta_array)).dot(f_delta_matrix))
                +np.diag(np.cos(delta_array)*I_x_array+np.sin(delta_array)*I_y_array))

        X8_matrix=-np.diag(X_d_array - X_d_p_array).dot(T_d0_p_inv). \
            dot((np.diag(np.sin(delta_array)).dot(e_q_matrix)) - (np.diag(np.cos(delta_array)).dot(f_q_matrix)))

        X9_matrix=-np.diag(X_d_array - X_d_p_array).dot(T_d0_p_inv). \
            dot((np.diag(np.sin(delta_array)).dot(e_d_matrix)) - (np.diag(np.cos(delta_array)).dot(f_d_matrix)))

        return X7_matrix,X8_matrix,X9_matrix

    def X10_to_12():
        T_q0_p_inv = np.diag(1 / T_q0_p_array)
        X10_matrix = np.diag(X_q_array - X_q_p_array).dot(T_q0_p_inv). \
            dot((np.diag(np.cos(delta_array)).dot(e_delta_matrix)) + (np.diag(np.sin(delta_array)).dot(f_delta_matrix))
                +np.diag(-np.sin(delta_array)*I_x_array+np.cos(delta_array)*I_y_array))

        X11_matrix = np.diag(X_q_array - X_q_p_array).dot(T_q0_p_inv). \
            dot((np.diag(np.cos(delta_array)).dot(e_q_matrix)) + (np.diag(np.sin(delta_array)).dot(f_q_matrix)))

        X12_matrix = np.diag(X_q_array - X_q_p_array).dot(T_q0_p_inv). \
                        dot((np.diag(np.cos(delta_array)).dot(e_d_matrix)) + (np.diag(np.sin(delta_array)).dot(f_d_matrix)))

        return X10_matrix,X11_matrix,X12_matrix

    def A00_mat():
        K1 = K1_matrix().copy()
        I = np.eye(num_G)*100*pi
        M_inv = np.diag(1/M_array)
        D = np.diag(D_array)
        T_d0_p_inv = np.diag(1/T_d0_p_array)
        X7=X7_to_9()[0]

        Mat10 = -np.dot(M_inv, K1)
        Mat11 = -np.dot(M_inv, D)

        A00 = subs_zeros_array(3, num_G, a01=I, a10=Mat10, a11=Mat11, a20=X7, a22=-T_d0_p_inv)
        return A00

    def A01_mat():
        K2 = K2_matrix().copy()
        K3 = K3_matrix().copy()
        M_inv = np.diag(1/M_array)
        X8,X9=X7_to_9()[1:]

        Mat11 = -np.dot(M_inv, K2)
        Mat12 = -np.dot(M_inv, K3)

        A01 = subs_zeros_array(3, num_G, a11=Mat11, a12=Mat12, a21=X8, a22=X9)
        return A01

    def A02_mat():
        T_d0_p_inv=np.diag(1/T_d0_p_array)
        return subs_zeros_array(3, num_G, a21=T_d0_p_inv)

    def A03_mat():
        M_inv=np.diag(1/M_array)
        return subs_zeros_array(3, num_G, a10=M_inv)

    def A10_mat():
        T_d0_pp_inv = np.diag(1/T_d0_pp_array)
        X1=X1_to_3()[0]
        X4=X4_to_6()[0]
        X10=X10_to_12()[0]

        A10 = subs_zeros_array(3, num_G, a00=X10, a10=X1, a12=T_d0_pp_inv, a20=X4)
        return A10

    def A11_mat():
        X2, X3 = X1_to_3()[1:]
        X5, X6 = X4_to_6()[1:]
        X11,X12= X10_to_12()[1:]

        T_q0_pp_inv = np.diag(1/T_q0_pp_array)
        T_q0_p_inv = np.diag(1/T_q0_p_array)

        A11 = subs_zeros_array(3, num_G, a00=-T_q0_p_inv, a01=X11, a02=X12, a11=X2, a12=X3, a20=T_q0_pp_inv, a21=X5, a22=X6)
        return A11

    def A20_21_24_mat():
        #AVR
        index_arr=data_AVR['index'].values
        type_arr=data_AVR['type'].values

        K1 = K1_matrix().copy()
        K2 = K2_matrix().copy()
        K3 = K3_matrix().copy()
        K4 = K4_matrix().copy()
        K5 = K5_matrix().copy()
        K6 = K6_matrix().copy()

        # I型（自并励）
        subs1 = K_A_array * (T1_array - T2_array) / (T1_array ** 2)
        subs2 = T2_array*(T3_array - T4_array) / (T1_array * (T3_array ** 2))
        subs3 = K_A_array * T2_array * T4_array / (T1_array * T3_array * T_E_array)
        # II型（他励）
        subs4 = K_A_array /T_A_array

        #I型（自并励）
        D00_I = np.diag(-subs1).dot(K4) \
              + np.diag(subs1*(a_array ** 2) * K_PSS_array * K_P_array).dot(K1)
        D01_I = np.diag(subs1*(a_array ** 2) * K_PSS_array * K_w_array * subs1)
        D04_I = np.diag(-subs1).dot(K5) \
              + np.diag(subs1*(a_array ** 2) * K_PSS_array * K_P_array).dot(K2)
        D05_I = np.diag(-subs1).dot(K6) \
              + np.diag(subs1*(a_array ** 2) * K_PSS_array * K_P_array).dot(K3)
        D06_I = np.diag(subs1*a_array**2)
        D07_I = np.diag(subs1*a_array)
        D08_I = np.diag(subs1)

        D10_I = np.diag(-subs2).dot(K4) \
              + np.diag(subs2*(a_array ** 2) * K_PSS_array * K_P_array).dot(K1)
        D11_I = np.diag(subs2*(a_array ** 2) * K_PSS_array * K_w_array * subs2)
        D14_I = np.diag(-subs2).dot(K5) \
              + np.diag(subs2*(a_array ** 2) * K_PSS_array * K_P_array).dot(K2)
        D15_I = np.diag(-subs2).dot(K6) \
              + np.diag(subs2*(a_array ** 2) * K_PSS_array * K_P_array).dot(K3)
        D16_I = np.diag(subs2 * a_array ** 2)
        D17_I = np.diag(subs2 * a_array)
        D18_I = np.diag(subs2)

        D20_I = np.diag(-subs3).dot(K4) \
              + np.diag(subs3*(a_array ** 2) * K_PSS_array * K_P_array).dot(K1)
        D21_I = np.diag(subs3*(a_array ** 2) * K_PSS_array * K_w_array * subs3)
        D24_I = np.diag(-subs3).dot(K5) \
              + np.diag(subs3*(a_array ** 2) * K_PSS_array * K_P_array).dot(K2)
        D25_I = np.diag(-subs3).dot(K6) \
              + np.diag(subs3*(a_array ** 2) * K_PSS_array * K_P_array).dot(K3)
        D26_I = np.diag(subs3 * a_array ** 2)
        D27_I = np.diag(subs3 * a_array)
        D28_I = np.diag(subs3)

        # II型（他励）
        D00_II = np.diag(-subs4).dot(K4) \
                + np.diag((a_array ** 2) * K_PSS_array * K_P_array).dot(K1)
        D01_II = np.diag((a_array ** 2) * K_PSS_array * K_w_array * subs4)
        D04_II = np.diag(-subs4).dot(K5) \
                + np.diag((a_array ** 2) * K_PSS_array * K_P_array).dot(K2)
        D05_II = np.diag(-subs4).dot(K6) \
                + np.diag((a_array ** 2) * K_PSS_array * K_P_array).dot(K3)
        D06_II = np.diag(subs4 * a_array ** 2)
        D07_II = np.diag(subs4 * a_array)
        D08_II = np.diag(subs4)

        #为了使得I型、II型中间状态量都是Ef，调整了顺序
        A20_mat_I = subs_zeros_array(3, num_G, a00=D00_I, a01=D01_I, a10=D20_I, a11=D21_I, a20=D10_I, a21=D11_I)
        A21_mat_I = subs_zeros_array(3, num_G, a01=D04_I, a02=D05_I, a11=D24_I, a12=D25_I, a21=D14_I, a22=D15_I)
        A24_mat_I = subs_zeros_array(3, num_G, a00=D06_I, a02=D08_I, a01=D07_I, a10=D26_I, a11=D27_I, a12=D28_I,
                                   a20=D16_I, a21=D17_I, a22=D18_I)
        A20_mat_II = subs_zeros_array(3, num_G, a00=D00_II, a01=D01_II)
        A21_mat_II = subs_zeros_array(3, num_G, a01=D04_II, a02=D05_II)
        A24_mat_II = subs_zeros_array(3, num_G, a00=D06_II, a01=D07_II, a02=D08_II)

        # # 根据type组成新矩阵
        # A20_mat,A21_mat,A24_mat=A20_mat_I.copy(),A21_mat_I.copy(),A24_mat_I.copy()
        # for i in range(type_arr.shape[0]):
        #     if type_arr[i]==2:
        #         index=index_arr[i]-1
        #         A20_mat[index*3:(index+1)*3],A21_mat[index*3:(index+1)*3],A24_mat[index*3:(index+1)*3]=\
        #             A20_mat_II[index*3:(index+1)*3].copy(),A21_mat_II[index*3:(index+1)*3].copy(),A24_mat_II[index*3:(index+1)*3].copy()
        #         # A20_mat[:,index*3:(index+1)*3],A21_mat[:,index*3:(index+1)*3],A24_mat[:,index*3:(index+1)*3]=\
        #         #     A20_mat_II[:,index*3:(index+1)*3].copy(),A21_mat_II[:,index*3:(index+1)*3].copy(),A24_mat_II[:,index*3:(index+1)*3].copy()

        A20_mat, A21_mat, A24_mat = np.zeros(A20_mat_I.shape),np.zeros(A21_mat_I.shape),np.zeros(A24_mat_I.shape)
        for i in range(type_arr.shape[0]):
            if type_arr[i]==1:
                index=index_arr[i]-1
                A20_mat[index::num_G], A21_mat[index::num_G], A24_mat[index::num_G]\
                    =A20_mat_I[index::num_G],A21_mat_I[index::num_G],A24_mat_I[index::num_G]
            if type_arr[i]==2:
                index=index_arr[i]-1
                A20_mat[index::num_G], A21_mat[index::num_G], A24_mat[index::num_G]\
                    =A20_mat_II[index::num_G],A21_mat_II[index::num_G],A24_mat_II[index::num_G]
        return A20_mat,A21_mat,A24_mat

    def A22_mat():
        #AVR
        index_arr=data_AVR['index'].values
        type_arr=data_AVR['type'].values

        # I型（自并励）
        A00_I = np.diag(1 / T1_array)
        A10_I = np.diag(T4_array / (T3_array * T_E_array))
        A11_I = np.diag(-1 / T_E_array)
        A12_I = np.diag(K_A_array / T_E_array)
        A20_I = np.diag((T3_array - T4_array) / ((T3_array ** 2) * K_A_array))
        A22_I = np.diag(-1 / T3_array)

        # II型（他励）
        A11_II= np.diag(-1/T_A_array)
        A13_II= np.diag(-K_A_array/T_A_array)
        A21_II= np.diag(1/T_E_array)
        A22_II=-A21_II.copy()
        A31_II=np.diag(K_F_array/(T_F_array*T_E_array))
        A32_II=-A31_II.copy()
        A33_II=np.diag(-1/T_F_array)

        A22_matrix_I = subs_zeros_array(3, num_G, a00=A00_I, a10=A10_I, a11=A11_I, a12=A12_I, a20=A20_I, a22=A22_I)
        A22_matrix_II = subs_zeros_array(3, num_G, a00=A11_II, a02=A13_II, a10=A21_II, a11=A22_II,
                                         a20=A31_II, a21=A32_II, a22=A33_II)

        # 根据type组成新矩阵
        A22_matrix=np.zeros(A22_matrix_I.shape)
        for i in range(type_arr.shape[0]):
            if type_arr[i] == 1:
                index = index_arr[i] - 1
                A22_matrix[index::num_G]=A22_matrix_I[index::num_G]
            if type_arr[i] == 2:
                index = index_arr[i] - 1
                A22_matrix[index::num_G]=A22_matrix_II[index::num_G]

        return A22_matrix

    # 该函数实现功能：合并两个含有nan的array，某位置只要在其中一个array中不是nan，则输出值不是nan
    # 两个array的数值位置没有重复
    def comb_nan_array(A_arr, B_arr):
        isnan_A, isnan_B = ~np.isnan(A_arr), ~np.isnan(B_arr)
        B_arr_copy = B_arr.copy()
        A_arr_copy = A_arr.copy()
        B_arr_copy[isnan_A] = 0
        A_arr_copy[isnan_B] = 0
        res_arr = B_arr_copy + A_arr_copy
        return res_arr

    def A30_mat():
        #汽轮机
        E11_ST = alpha_array*K_mH_array_ST*T_W_array_ST*K_delta_array_ST/(T_S_array_ST*T0_array_ST)
        E21_ST = -K_delta_array_ST/T_S_array_ST
        E31_ST = K_mH_array_ST*T_W_array_ST*K_delta_array_ST/(T_S_array_ST*T0_array_ST)

        #水轮机
        E11_WT = K_mH_array_WT * T_W_array_WT * K_delta_array_WT / (T_S_array_WT * T0_array_WT)
        E21_WT = -K_delta_array_WT / T_S_array_WT
        E31_WT = -K_delta_array_WT*K_beta_array/T_S_array_WT


        E11 = np.diag(comb_nan_array(E11_ST,E11_WT))
        E21 = np.diag(comb_nan_array(E21_ST, E21_WT))
        E31 = np.diag(comb_nan_array(E31_ST, E31_WT))

        A30_matrix = subs_zeros_array(3, num_G, a01=E11, a11=E21, a21=E31)

        return A30_matrix


    def A33_mat():
        # 汽轮机
        B11_ST = -1/T_RH_array
        B12_ST = alpha_array*K_mH_array_ST/T0_array_ST*(1+T_W_array_ST*K_i_array_ST/T_S_array_ST)
        B13_ST = 1/T_RH_array - alpha_array/T0_array_ST
        B22_ST = -1/T_S_array_ST
        B23_ST = T_RH_array*0
        B32_ST = K_mH_array_ST/T0_array_ST*(1+T_W_array_ST*K_i_array_ST/T_S_array_ST)
        B33_ST = -1/T0_array_ST

        # 水轮机
        B11_WT = -1/T0_array_WT
        B12_WT = K_mH_array_WT/T0_array_WT*(1+T_W_array_WT*K_i_array_WT/T_S_array_WT)
        B13_WT = K_mH_array_WT*T_W_array_WT/(T0_array_WT*T_S_array_WT)
        B22_WT = -K_i_array_WT/T_S_array_WT
        B23_WT = -1/T_S_array_WT
        B32_WT = -K_beta_array*K_i_array_WT/T_S_array_WT
        B33_WT = -(K_beta_array/T_S_array_WT+1/T_i_array)

        B11 = np.diag(comb_nan_array(B11_ST, B11_WT))
        B12 = np.diag(comb_nan_array(B12_ST, B12_WT))
        B13 = np.diag(comb_nan_array(B13_ST, B13_WT))
        B22 = np.diag(comb_nan_array(B22_ST, B22_WT))
        B23 = np.diag(comb_nan_array(B23_ST, B23_WT))
        B32 = np.diag(comb_nan_array(B32_ST, B32_WT))
        B33 = np.diag(comb_nan_array(B33_ST, B33_WT))

        A33_matrix = subs_zeros_array(3, num_G, a00=B11, a01=B12, a02=B13, a11=B22, a12=B23, a21=B32, a22=B33)
        return A33_matrix

    def A40_mat():
        F12=np.diag(K_PSS_array*K_w_array/T_w_array)
        F22=np.diag(K_PSS_array*K_w_array*(1-a_array)/T2_PSS_array)
        F32=np.diag(K_PSS_array*K_w_array*(1-a_array)*a_array/T2_PSS_array)

        A40_matrix = subs_zeros_array(3, num_G, a01=F12, a11=F22, a21=F32)
        return A40_matrix

    def A44_mat():
        C11=np.diag(-1/T_w_array)
        C21=np.diag(-(1-a_array)/T2_PSS_array)
        C22=np.diag(-1/T2_PSS_array)
        C31=np.diag(-a_array*(1-a_array)/T2_PSS_array)
        C32=np.diag((1-a_array)/T2_PSS_array)
        C33=np.diag(-1/T2_PSS_array)

        A44_matrix = subs_zeros_array(3, num_G, a00=C11, a10=C21, a11=C22, a20=C31, a21=C32, a22=C33)
        return A44_matrix

    # 不含PSS模型的A矩阵
    def A_mat():
        A00 = A00_mat().copy()
        A01 = A01_mat().copy()
        A02 = A02_mat().copy()
        A03 = A03_mat().copy()
        A10 = A10_mat().copy()
        A11 = A11_mat().copy()
        A20,A21,A24 = A20_21_24_mat()
        A22 = A22_mat().copy()
        A30 = A30_mat().copy()
        A33 = A33_mat().copy()
        # A40= A40_mat().copy()
        # A44= A44_mat().copy()

        A_matrix = subs_zeros_array(4, 3 * num_G, a00=A00, a01=A01, a02=A02, a03=A03, a10=A10, a11=A11, a20=A20,
                                    a21=A21, a22=A22, a30=A30, a33=A33,
                                    # a40=A40, a44=A44,a24=A24
                                    )

        # 删去含有nan的行与列
        # 找出励磁/原动机中含有nan的对应元素序号
        nan_index_PM = sorted(np.argwhere(np.isnan(data_PM['调差系数'].values) == 1)[:, 0], reverse=True)  # 降序
        nan_index_AVR = sorted(np.argwhere(np.isnan(data_AVR['K_A'].values) == 1)[:, 0], reverse=True)
        # nan_index_PSS = sorted(np.argwhere(data_PSS['K_PSS'].values ==0 )[:, 0], reverse=True)

        # for i in range(3):  # 3阶
        #     for j in nan_index_PSS:
        #         A_matrix = np.delete(A_matrix, num_G * (14 - i) + j, 0)#8是因为共有12个状态量，调速器在倒数第六-倒数第四
        #         A_matrix = np.delete(A_matrix, num_G * (14 - i) + j, 1)

        for i in range(3):  # 3阶
            for j in nan_index_PM:
                A_matrix = np.delete(A_matrix, num_G * (11 - i) + j, 0) #11是因为共有12个状态量，调速器在倒数三个
                A_matrix = np.delete(A_matrix, num_G * (11 - i) + j, 1)

        for i in range(3):  # 3阶
            for j in nan_index_AVR:
                A_matrix = np.delete(A_matrix, num_G * (8 - i) + j, 0)#8是因为共有12个状态量，调速器在倒数第六-倒数第四
                A_matrix = np.delete(A_matrix, num_G * (8 - i) + j, 1)

        val = get_UV(A_matrix)[0]
        # print(val[val.real > 0])
        return A_matrix

    return A_mat()

# Tw=1
# Kw=1500
A_mat=A_Generation(Power_flow,Y,bus_data,data_G,data_AVR,data_PM)
# A_mat=np.insert(A_mat,-1, np.array([0,0,-Kw/Tw]+[0]*84),axis=0)
# A_mat=np.insert(A_mat,-1, np.array([0]*49+[1]+[0]*37+[-1/Tw]),axis=1)
# pd.DataFrame(A_mat).to_excel('E:\文件\电力系统分析\暂态稳定分析\data_python.xlsx',index=None)
# print(np.argwhere(np.isnan(A_mat)))




def correlation_factors(U, V):
    """
    维度n*n，n个状态量和特征根，衡量两者两两相关性
    parameters
    ==========
    U A矩阵右特征向量构成矩阵[u1,u2,...,un]
    V A矩阵左特征向量构成矩阵[v1,v2,...,vn]
    """
    P = U * V / np.diag(V.T.dot(U))
    return (P.real**2+P.imag**2)**0.5

val=get_UV(A_mat)[0]
U,V=get_UV(A_mat)[1],get_UV(A_mat)[2]
# print(val)
# print(val[val.real>0])
# print(correlation_factors(U, V)[24:26])

vals_real=val.real
vals_imag=val.imag



#作图
# plt.plot(vals_real,vals_imag,'o')
# plt.ylabel('imag',verticalalignment='top')
# plt.xlabel('real',horizontalalignment='right')
# ax = plt.gca()                                            # get current axis 获得坐标轴对象

# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')         # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边

# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')          # 指定下边的边作为 x 轴   指定左边的边为 y 轴

# ax.spines['bottom'].set_position(('data', 0))   #指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
# ax.spines['left'].set_position(('data', 0))
# plt.show()
