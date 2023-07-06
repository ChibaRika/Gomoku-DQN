import torch
from torch import nn
import numpy as np
import copy
import os

import random
from graphics import *
import pyautogui

#棋盘行数
x = 15

#棋盘列数
y = 15

#单次训练局数（更新网络局数）
size = 10

#是否显示自对弈窗口
windows_visible = False

#状态空间s（每一手棋盘的记录）
s = np.zeros((x*y,x,y), dtype = int)

#动作空间a（当前落子位置的记录）
a = np.zeros((x*y,2), dtype = int)

totalmove = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("devide:",device)

#变量初始化
def variable_initialization():
    
    global move,chessboard,s,a,bwin,wwin,draw,avgloss
    
    #初始化棋盘 __=0 ○=-1 ●=1
    chessboard = np.zeros((x,y), dtype = int)
    move = 0
    
    bwin = False
    wwin = False
    draw = False

    avgloss = 0
    
#GUI窗口
def windows():
    
    global win

    #绘制窗口
    win = GraphWin("Gomoku",1200,1200)
    win.setBackground(color_rgb(227,188,116))
    
    linex = liney = 75

    #绘制棋盘网格
    while linex < 1126:
        line = Line(Point(linex,75),Point(linex,1126))
        line.draw(win)
        linex += 75
        
    while liney < 1126:
        line = Line(Point(75,liney),Point(1126,liney))
        line.draw(win)
        liney += 75

    #绘制天元
    tianyuan = Circle(Point(7 * 75 + 75, 7 * 75 + 75),8)
    tianyuan.setFill("black")
    tianyuan.draw(win)
    
    #绘制星位
    xingwei = Circle(Point(3 * 75 + 75, 3 * 75 + 75),8)
    xingwei.setFill("black")
    xingwei.draw(win)
    
    xingwei = Circle(Point(3 * 75 + 75, 11 * 75 + 75),8)
    xingwei.setFill("black")
    xingwei.draw(win)
    
    xingwei = Circle(Point(11 * 75 + 75, 3 * 75 + 75),8)
    xingwei.setFill("black")
    xingwei.draw(win)
    
    xingwei = Circle(Point(11 * 75 + 75, 11 * 75 + 75),8)
    xingwei.setFill("black")
    xingwei.draw(win)
    
def aimove():
    
    global win,move,totalmove,chessboard,newpiece,x,y,windows_visible
    
    values = forward(model,chessboard)
    values = values.cpu().detach().numpy()

    #随机化每步
    seed = random.randint(1,10)
    
    if move % 2 == 0 and seed <= 5:
        
        for i in range(225):
        
            posx = np.argmax(values) // 15
            posy = np.argmax(values) % 15
            
            if chessboard[posx][posy] != 0:
                values[np.argmax(values)] = -1
            else:
                break

        chessboard[posx][posy] = 1
        if windows_visible == True:
            piece = Circle(Point(posx * 75 + 75, posy * 75 + 75),30)
            piece.setFill("black")
        
    if move % 2 == 0 and seed > 5:
        
        for i in range(225):
            
            posx = random.randint(0,14)
            posy = random.randint(0,14)
            
            if chessboard[posx][posy] == 0:
                break
            
        chessboard[posx][posy] = 1
        if windows_visible == True:
            piece = Circle(Point(posx * 75 + 75, posy * 75 + 75),30)
            piece.setFill("black")
            
    if move % 2 == 1 and seed <= 5:
        
        for i in range(225):
        
            posx = np.argmin(values) // 15
            posy = np.argmin(values) % 15
            
            if chessboard[posx][posy] != 0:
                values[np.argmin(values)] = 1
            else:
                break

        chessboard[posx][posy] = -1
        if windows_visible == True:
            piece = Circle(Point(posx * 75 + 75, posy * 75 + 75),30)
            piece.setFill("white")
        
    if move % 2 == 1 and seed > 5:
        
        for i in range(225):
            
            posx = random.randint(0,14)
            posy = random.randint(0,14)
            
            if chessboard[posx][posy] == 0:
                break
            
        chessboard[posx][posy] = -1
        if windows_visible == True:
            piece = Circle(Point(posx * 75 + 75, posy * 75 + 75),30)
            piece.setFill("white")

    #记录s,a
    for i in range(x):
        
        for j in range(y):
            
            s[totalmove][i][j] = chessboard[i][j]
            a[totalmove][0] = posx
            a[totalmove][1] = posy
            
    """
    if move % 2 == 0:
        print(values[np.argmax(values)])
        
    if move % 2 == 1:
        print(values[np.argmin(values)])
    """
    
    if move % 2 == 0:
        winjudgement(1,posx,posy)
        
    if move % 2 == 1:
        winjudgement(-1,posx,posy)
        
    #绘制棋子
    if windows_visible == True:
        piece.draw(win)
    
    #手数+1
    move += 1
    totalmove += 1
    
    #如果存在,删除上一手棋子高亮
    if windows_visible == True:
        if "newpiece" in globals():
            newpiece.undraw()
        
        newpiece = Polygon(Point(posx * 75 + 75, posy * 75 + 75), \
                           Point(posx * 75 + 75 + 30, posy * 75 + 75), \
                           Point(posx * 75 + 75, posy * 75 + 75 + 30))
        newpiece.setOutline("red")
        newpiece.setFill("red")
        
    #绘制上一手棋子高亮
    if windows_visible == True:
        newpiece.draw(win)
        
#胜利判断
def winjudgement(chess,movex,movey):

    global move,bwin,wwin,draw

    for i in range(5): #横向

        if movey-4+i < 0:
            continue

        if movey+i > 14:
            break
            
        if chessboard[movex][movey-4+i] == chess and \
            chessboard[movex][movey-3+i] == chess and \
            chessboard[movex][movey-2+i] == chess and \
            chessboard[movex][movey-1+i] == chess and \
            chessboard[movex][movey+i] == chess:
            if chess == 1:
                bwin = True
                #print("Black Win!")
                return
            if chess == -1:
                wwin = True
                #print("White Win!")
                return
            
    for i in range(5): #纵向

        if movex-4+i < 0:
            continue

        if movex+i > 14:
            break

        if chessboard[movex-4+i][movey] == chess and \
            chessboard[movex-3+i][movey] == chess and \
            chessboard[movex-2+i][movey] == chess and \
            chessboard[movex-1+i][movey] == chess and \
            chessboard[movex+i][movey] == chess:
            if chess == 1:
                bwin = True
                #print("Black Win!")
                return
            if chess == -1:
                wwin = True
                #print("White Win!")
                return

    for i in range(5): #斜向，左上到右下

        if movex-4+i < 0 or movey-4+i < 0:
            continue

        if movex+i > 14 or movey+i > 14:
            break
        
        if chessboard[movex-4+i][movey-4+i] == chess and \
            chessboard[movex-3+i][movey-3+i] == chess and \
            chessboard[movex-2+i][movey-2+i] == chess and \
            chessboard[movex-1+i][movey-1+i] == chess and \
            chessboard[movex+i][movey+i] == chess:
            if chess == 1:
                bwin = True
                #print("Black Win!")
                return
            if chess == -1:
                wwin = True
                #print("White Win!")
                return

    for i in range(5): #斜向，右上到左下

        if movex-4+i < 0 or movey+4-i > 14:
            continue

        if movex+i > 14 or movey-i < 0:
            break
        
        if chessboard[movex-4+i][movey+4-i] == chess and \
            chessboard[movex-3+i][movey+3-i] == chess and \
            chessboard[movex-2+i][movey+2-i] == chess and \
            chessboard[movex-1+i][movey+1-i] == chess and \
            chessboard[movex+i][movey-i] == chess:
            if chess == 1:
                bwin = True
                #print("Black Win!")
                return
            if chess == -1:
                wwin = True
                #print("White Win!")
                return
            
    if move == 225:
        draw = True
        #print("Draw!")
        
def play():
    
    global bwin,wwin,draw
    
    for i in range(225):
        aimove()
        if bwin == True or wwin == True or draw == True:
            return
        
#模型初始化
def initialization():
    
    input_size = 225
    hidden_sizes = [256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
    output_size = 225
    
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[2], hidden_sizes[3]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[3], hidden_sizes[4]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[4], hidden_sizes[5]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[5], hidden_sizes[6]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[6], hidden_sizes[7]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[7], hidden_sizes[8]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[8], hidden_sizes[9]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[9], output_size),
                          nn.Tanh())
    
    print(model)
    print(model[0].weight)
    print(model[2].weight)
    print(model[4].weight)
    print(model[6].weight)
    print(model[8].weight)
    print(model[10].weight)
    print(model[12].weight)
    print(model[14].weight)
    print(model[16].weight)
    print(model[18].weight)
    print(model[20].weight)
    
    #保存模型
    torch.save(model, "model.pth")
    
#前向传播
def forward(network,inputs):

    inputs = inputs.flatten()
    inputs = inputs.astype('float32')
    inputs_tensor = torch.from_numpy(inputs)
    inputs_tensor = inputs_tensor.to(device)
    outputs = network(inputs_tensor)
    return outputs

if os.path.isfile("model.pth") == False:
    initialization()
    
#加载模型
model = torch.load("model.pth")
model = model.to(device)

#目标值
#targets = np.array([1], dtype='float32')
#targets_tensor = torch.from_numpy(targets)

#损失函数，MSELoss为均方误差损失
loss_function = nn.MSELoss()

#损失函数，L1Loss为范数损失
#loss_function = nn.L1Loss()

#SGD（随机梯度下降法）优化器，lr为学习率
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#AdamW优化器
#optimizer = torch.optim.AdamW(model.parameters())

def train():

    global targets,model_copy,bwin,wwin,draw,avgloss

    #复制模型用于训练
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)

    #AdamW优化器
    optimizer = torch.optim.AdamW(model_copy.parameters())
    
    #打乱数据索引
    shufflelist = np.random.permutation(np.arange(0,s.shape[0]))

    #使用下一步进行训练（最后一步不进行训练）
    for i in range(s.shape[0] - 1):

        #最后一步
        if shufflelist[i] == s.shape[0] - 1:
            continue
        
        inputs = s[shufflelist[i],:,:]
        inputs = inputs.flatten()
        inputs = inputs.astype('float32')
        inputs_tensor = torch.from_numpy(inputs)

        #下一步的奖励r
        #黑胜，前两步的白棋负奖励
        if bwin == True and shufflelist[i] == s.shape[0] - 3:
            reward = -0.9
        #黑胜，前一步的黑棋正奖励
        if bwin == True and shufflelist[i] == s.shape[0] - 2:
            reward = 1
        #白胜，前两步的黑棋负奖励
        if wwin == True and shufflelist[i] == s.shape[0] - 3:
            reward = -0.9
        #白胜，前一步的白棋正奖励
        if wwin == True and shufflelist[i] == s.shape[0] - 2:
            reward = 1
        else:
            reward = 0
            
        #惩罚系数γ
        gamma = -0.9

        #目标
        targets = torch.max(forward(model,s[shufflelist[i]+1,:,:])) * gamma + reward
        
        #计算损失
        outputs = forward(model_copy,inputs)
        loss = loss_function(outputs,targets)
        
        #优化器调节梯度为0
        optimizer.zero_grad()
        
        #损失反向传播
        loss.backward()
        
        #对每个参数进行调优
        optimizer.step()
        
        #print("loss:",loss)
        avgloss += float(loss)

#训练次数
for k in range(10):
    
    avgloss = 0
    
    model = torch.load("model.pth")
    model = model.to(device)
    
    for m in range(size):
        
        #状态空间s（每一手棋盘的记录）
        s = np.zeros((x*y,x,y), dtype = int)
    
        #动作空间a（当前落子位置的记录）
        a = np.zeros((x*y,2), dtype = int)
    
        totalmove = 0
        
        for i in range(1):
            if windows_visible == True:
                windows()
            variable_initialization()
            play()
            
        #删除数据为0的部分
        for i in range(225):
            
            if np.any(s[i,:,:]) == False:
                s = s[:i,:,:]
                a = a[:i,:]
                break
            
        for i in range(1):
            train()
            
        print("training is finished")
        
        avgloss /= s.shape[0] - 1
        
        torch.save(model_copy, "model.pth")
        
    if avgloss >= 0.0001:
        print("avgloss:","%0.6f"%avgloss)
    else:
        print("avgloss:","%0.6e"%avgloss)
        
