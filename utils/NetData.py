import csv
import numpy as np
FilePath_Node = 'data/Bus_data.csv'#Bus_data为节点数据
FilePath_Line = 'data/B_data.csv'#B_data为支路数据
def GetNumNode(FilePath_Node):
    csv_file = open(FilePath_Node)
    lines = csv.reader(csv_file)
    NumNode = 0
    for line in lines:
        NumNode = NumNode + 1  # 得到节点数目
    # print(NumNode)
    return (NumNode)

#GetNumNode(FilePath_Node)

def GetNodeData(FilePath_Node,*NumNode):
    if not NumNode:  # 空的
        NumNode = GetNumNode(FilePath_Node)
    Bus_data = [] # 初始化
    i = 0
    csv_file = open(FilePath_Node)
    lines = csv.reader(csv_file)
    for line in lines:
        for j in range(0,10):
            Bus_data.append(line[j])
        i = i+1
    Bus_data = np.array(Bus_data).reshape(-1, 10)
    return(Bus_data)

#Bus_data = GetNodeData(FilePath_Node)
#print(Bus_data)

def GetNumLine(FilePath_Line):
    csv_file = open(FilePath_Node)
    lines = csv.reader(csv_file)
    NumLine = 0
    for line in lines:
        NumLine = NumLine+1  # 得到节点数目
    return(NumLine)

#GetNumNode(FilePath_Line)

def GetLineData(FilePath_Node,*NumLine):
    if not NumLine:  # 空的
        NumLine = GetNumLine(FilePath_Node)
    B_data = [] # 初始化
    i = 0
    csv_file = open(FilePath_Node)
    lines = csv.reader(csv_file)
    for line in lines:
        for j in range(0,6):
            B_data.append(line[j])
        i = i+1
    B_data = np.array(B_data).reshape(-1, 6)
    return(B_data)

#B_data = GetLineData(FilePath_Line)
#print(B_data)