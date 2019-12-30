import numpy as np
from .NetData import GetNodeData, GetLineData
import pandas as pd

def GetY(Bus_data,B_data):#Bus_data为节点数据，B_data为支路数据
    NumNode = Bus_data.shape[0]
    NumLine = B_data.shape[0]
    Y = np.zeros([NumNode, NumNode]) + np.zeros([NumNode, NumNode]) * 1j
    G0 = Bus_data[:, 8]  # 节点对地电导
    B0 = Bus_data[:, 9]  # 节点对地电纳
    # 开始节点导纳矩阵的建立
    for i in range(NumLine):
        Node1 = int(int(B_data[i, 0]) - 1)
        Node2 = int(int(B_data[i, 1]) - 1)
        # print(Node1,Node2)
        R = float(B_data[i, 2])
        X = float(B_data[i, 3])
        if float(B_data[i, 5]) == 0:  # 普通线路，无变压器
            B_2 = float(B_data[i, 4]) / 2
            Y[Node1, Node1] = Y[Node1, Node1] + B_2 * 1j + 1 / (R + 1j * X)
            Y[Node2, Node2] = Y[Node2, Node2] + B_2 * 1j + 1 / (R + 1j * X)
            Y[Node1, Node2] = Y[Node1, Node2] - 1 / (R + 1j * X)
            Y[Node2, Node1] = Y[Node2, Node1] - 1 / (R + 1j * X)
        else:  # 有变压器支路
            K = float(B_data[i, 5])
            YT = 1 / (R + 1j * X)
            Y[Node2, Node2] = Y[Node2, Node2] + (K - 1) / K * YT + YT / K
            Y[Node1, Node1] = Y[Node1, Node1] + (1 - K) / K ** 2 * YT + YT / K
            Y[Node1, Node2] = Y[Node1, Node2] - 1 / K * YT
            Y[Node2, Node1] = Y[Node2, Node1] - 1 / K * YT
    for i in range(NumNode):
        Node = int(int(Bus_data[i,0])-1)   # 第一列为节点编号
        Y[Node,Node] = Y[Node,Node]+float(G0[i])+1j*float(B0[i])
    # 节点导纳矩阵创建完毕
    return (Y)

FilePath_Node = 'data/Bus_data.csv'
FilePath_Line = 'data/B_data.csv'

NodeData = GetNodeData(FilePath_Node)
LineData = GetLineData(FilePath_Line)
Y = GetY(NodeData,LineData)
# print(Y)
#Y_1 = pd.DataFrame(Y)
#Y_1.to_csv("YData.csv")