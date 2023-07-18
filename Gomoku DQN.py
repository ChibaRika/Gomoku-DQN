import torch
from torch import nn
import numpy as np
import copy
import os
import random
import matplotlib.pylab as plt

from graphics import *
import pyautogui

#棋盘行数
x = 15

#棋盘列数
y = 15

#单次训练局数（更新网络局数）
size = 10

#更新网络次数
upgrade = 10

avgloss = np.zeros((upgrade))
avgstep = np.zeros((upgrade))

#是否显示自对弈窗口
windows_visible = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

#变量初始化
def variable_initialization():
    
    global move,chessboard,s,a,bwin,wwin,draw
    
    #初始化棋盘 __=0 ○=-1 ●=1
    chessboard = np.zeros((x,y), dtype = int)
    move = 0
    
    bwin = False
    wwin = False
    draw = False
    
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
    
    global win,move,chessboard,newpiece,x,y,windows_visible,maxvalue,maxposx,maxposy
    
    firstvalue = True
    
    #epsilon-greedy策略
    seed = random.randint(1,10)
    
    for i in range(x):
        for j in range(y):
            
            if chessboard[i][j] != 0:
                continue
            
            if move % 2 == 0:
                chessboard[i][j] = 1
            if move % 2 == 1:
                chessboard[i][j] = -1
                
            values = model.forward(chessboard)
            values = values.cpu().detach().numpy()
            
            if move % 2 == 1:
                values *= -1
                
            if firstvalue == True:
                maxvalue[move] = values
                firstvalue = False
                
            #寻找最大价值位置落子
            if values >= maxvalue[move]:
                maxvalue[move] = values
                maxposx[move] = i
                maxposy[move] = j
                
            chessboard[i][j] = 0
            
    if seed <= 3:
        
        for i in range(100000):
            
            posx = random.randint(0,x-1)
            posy = random.randint(0,y-1)
            
            if chessboard[posx][posy] == 0:
                print("random step")
                break
            
    if seed > 3:
        
            posx = maxposx[move]
            posy = maxposy[move]
            
    if move % 2 == 0:
        
        chessboard[posx][posy] = 1
        
        if windows_visible == True:
            piece = Circle(Point(posx * 75 + 75, posy * 75 + 75),30)
            piece.setFill("black")
            
    if move % 2 == 1:
        
        chessboard[posx][posy] = -1
        
        if windows_visible == True:
            piece = Circle(Point(posx * 75 + 75, posy * 75 + 75),30)
            piece.setFill("white")
            
    #记录s,a
    for i in range(x):
        
        for j in range(y):
            
            s[move][i][j] = chessboard[i][j]
            a[move][0] = posx
            a[move][1] = posy
            
    print("value:",maxvalue[move])
    
    if move % 2 == 0:
        winjudgement(1,posx,posy)
        
    if move % 2 == 1:
        winjudgement(-1,posx,posy)
        
    #绘制棋子
    if windows_visible == True:
        piece.draw(win)
        
    #手数+1
    move += 1
    
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
        
class net(nn.Module):
    
    def __init__(self):
        super(net,self).__init__()
        
        #输入模块
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=256, stride=1, kernel_size=5, padding=2)
        
        self.layer2 = torch.nn.GroupNorm(32,256)
        
        self.layer3 = nn.ReLU()
        
        #残差模块1
        self.layer4 = nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=5, padding=2)
        
        self.layer5 = torch.nn.GroupNorm(32,256)
        
        self.layer6 = nn.ReLU()
        
        self.layer7 = nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=5, padding=2)
        
        self.layer8 = torch.nn.GroupNorm(32,256)
        
        #残差模块2
        self.layer9 = nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=5, padding=2)
        
        self.layer10 = torch.nn.GroupNorm(32,256)
        
        self.layer11 = nn.ReLU()
        
        self.layer12 = nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=5, padding=2)
        
        self.layer13 = torch.nn.GroupNorm(32,256)
        
        
        self.layer14 = nn.ReLU()
        
        
        #价值头
        self.layer15 = nn.Conv2d(in_channels=256, out_channels=1, stride=1, kernel_size=1, padding=0)
        
        self.layer16 = torch.nn.GroupNorm(1,1)
        
        self.layer17 = nn.ReLU()
        
        self.layer18 = nn.Flatten(start_dim=0, end_dim=3)
        
        self.layer19 = nn.Linear(in_features=225, out_features=128)
        
        self.layer20 = nn.ReLU()
        
        self.layer21 = nn.Linear(in_features=128, out_features=1)
        
        self.layer22 = nn.Tanh()
        
        #He正态分布初始化
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
                
    def forward(self,inputs):
        
        global x,y
        
        inputs = inputs.flatten()
        
        #给输入分通道[batch_size, channel, x, y]
        inputs_channels = np.zeros((1,3,x,y), dtype = int)
        
        for i in range(15):
            for j in range(15):
                
                if inputs[i * 15 + j] == 1:
                    inputs_channels[0][0][i][j] = 1 #黑棋
                    
                if inputs[i * 15 + j] == -1:
                    inputs_channels[0][1][i][j] = 1 #白棋
                    
                if inputs[i * 15 + j] == 0:
                    inputs_channels[0][2][i][j] = 1 #空
                    
        inputs_channels = inputs_channels.astype('float32')
        
        inputs_tensor = torch.from_numpy(inputs_channels)
        inputs_tensor = inputs_tensor.to(device)
        
        mid = self.layer1(inputs_tensor)
        mid = self.layer2(mid)
        mid = self.layer3(mid)
        mid = self.layer4(mid)
        mid = self.layer5(mid)
        mid = self.layer6(mid)
        mid = self.layer7(mid)
        mid = self.layer8(mid)
        mid = self.layer9(mid)
        mid = self.layer10(mid)
        mid = self.layer11(mid)
        mid = self.layer12(mid)
        mid = self.layer13(mid)
        mid = self.layer14(mid)
        mid = self.layer15(mid)
        mid = self.layer16(mid)
        mid = self.layer17(mid)
        mid = self.layer18(mid)
        mid = self.layer19(mid)
        mid = self.layer20(mid)
        mid = self.layer21(mid)
        outputs = self.layer22(mid)
        return outputs
    
#模型初始化
model = net()

if os.path.isfile("model.pth") == False:
    torch.save(model, "model.pth")
    
#加载模型
model = torch.load("model.pth")
model = model.to(device)

#损失函数，MSELoss为均方误差损失
loss_function = nn.MSELoss()

def train():
    
    global targets,model_copy,bwin,wwin,draw,totalloss,maxvalue
    
    #AdamW优化器
    optimizer = torch.optim.AdamW(model_copy.parameters())
    
    #打乱数据索引
    shufflelist = np.random.permutation(np.arange(0,s.shape[0]))
    
    #使用下一步进行训练
    for i in range(s.shape[0]):
        
        inputs = s[shufflelist[i],:,:]
        inputs = inputs.flatten()
        inputs = inputs.astype('float32')
        inputs_tensor = torch.from_numpy(inputs)
        
        #下一步的奖励r
        #黑胜，前一步的黑棋正奖励
        if bwin == True and shufflelist[i] == s.shape[0] - 1:
            reward = 1
        #白胜，前一步的白棋正奖励
        elif wwin == True and shufflelist[i] == s.shape[0] - 1:
            reward = 1
        else:
            reward = 0
            
        #惩罚系数γ
        gamma = -0.9
        
        #目标
        targets = np.array([0], dtype = 'float32')
        
        #黑胜，前两步的白棋负价值
        if bwin == True and shufflelist[i] == s.shape[0] - 2:
            targets[0] = -0.9
        #白胜，前两步的黑棋负价值
        elif wwin == True and shufflelist[i] == s.shape[0] - 2:
            targets[0] = -0.9
            
        #最后一步targets = reward
        elif shufflelist[i] == s.shape[0] - 1:
            targets[0] = reward
        else:
            targets[0] = maxvalue[shufflelist[i]+1] * gamma + reward
            
        targets = torch.from_numpy(targets).cuda()
        
        #计算损失
        outputs = model_copy.forward(inputs)
        
        if shufflelist[i] % 2 == 1:
            outputs = torch.mul(outputs,-1)
            
        loss = loss_function(outputs,targets)
        
        #print("outputs:",outputs)
        #print("targets:",targets)
        
        #优化器调节梯度为0
        optimizer.zero_grad()
        
        #损失反向传播
        loss.backward()
        
        #对每个参数进行调优
        optimizer.step()
        
        #print("loss:",loss)
        totalloss += float(loss)
        
for k in range(upgrade):
    
    totalloss = 0
    totalstep = 0
    
    model = torch.load("model.pth")
    model = model.to(device)
    
    #复制模型用于训练
    model_copy = net()
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    
    for m in range(size):
        
        #状态空间s（每一手棋盘的记录）
        s = np.zeros((x*y,x,y), dtype = int)
        
        #动作空间a（当前落子位置的记录）
        a = np.zeros((x*y,2), dtype = int)
        
        #最大价值
        maxvalue = np.zeros((x*y), dtype = 'float32')
        maxposx = np.zeros((x*y), dtype = 'int')
        maxposy = np.zeros((x*y), dtype = 'int')
        
        for i in range(1):
            if windows_visible == True:
                windows()
            variable_initialization()
            play()
            if windows_visible == True:
                win.close()
                
        #删除数据为0的部分
        for i in range(225):
            
            if np.any(s[i,:,:]) == False:
                s = s[:i,:,:]
                a = a[:i,:]
                maxvalue = maxvalue[:i]
                break
            
        for i in range(1):
            train()
            
        print()
        totalstep += s.shape[0]
        
    torch.save(model_copy, "model.pth")
    print("training is finished")
    print()
    
    avgloss[k] = totalloss / totalstep
    if avgloss[k] >= 0.0001:
        print("avgloss:","%0.6f"%avgloss[k])
    else:
        print("avgloss:","%0.6e"%avgloss[k])
        
    avgstep[k] = totalstep / size
    print("avgstep:","%0.4f"%avgstep[k])
    print()
    
plt.subplot(1, 2, 1)
plt.plot(range(0,upgrade * size,size),avgloss)
plt.yscale('log')
plt.xlabel("training times")
plt.ylabel("avgloss")

plt.subplot(1, 2, 2)
plt.plot(range(0,upgrade * size,size),avgstep)
plt.xlabel("training times")
plt.ylabel("avgstep")

plt.show()
