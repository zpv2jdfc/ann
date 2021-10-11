import numpy as np
import pylab as p
class ANN:
    def __init__(self,**arg):
        init = {'layer': [],
                'active_function': 'sigmoid',
                'output_function': 'softmax',
                'learning_rate': 1.5,
                'objective_function': 'crossEntropy'
                }
        param = dict() #字典结构实现参数列表
        param.update(init)
        param.update(arg)
        self.layer = param['layer']
        self.active_function = param['active_function']
        self.output_function = param['output_function']
        self.learning_rate = param['learning_rate']
        self.objective_function = param['objective_function']
        #参数
        self.w = dict(); self.b = dict(); self.theta=dict(); self.a=dict(); self.sw=dict();self.sb=dict();
        self.depth = len(self.layer)-1
        for i in range(1,self.depth+1):
            self.w[i] = np.matrix(np.random.rand(self.layer[i],self.layer[i-1])*2-1)
            self.b[i] = np.matrix(np.random.rand(self.layer[i],1) * 2 - 1)
def forward(nn,x):
    nn.a[0]=np.matrix(x).T;
    for i in range(1,nn.depth+1):
        nn.a[i] = np.dot(nn.w[i],nn.a[i-1]) + nn.b[i];
        if(i!=nn.depth):
            if(nn.active_function=='sigmoid'):
                nn.a[i]=1/(1+np.exp(-nn.a[i]))
    if(nn.output_function == 'softmax'):
        denominator = 0;#分母
        for i in range(0,nn.layer[len(nn.layer)-1]):
            denominator+=np.exp(nn.a[nn.depth][i])
        nn.a[nn.depth]=np.exp(nn.a[nn.depth])/denominator
       # for i in range(0, nn.layer[len(nn.layer) - 1]):
        #    nn.y[i] = np.exp(nn.a[nn.depth-1][i])/denominator
    return nn
def nn_backpropagation(nn,y):
    nn.theta[nn.depth]=(nn.a[nn.depth]-np.matrix(y).T)
    nn.sw[nn.depth]=nn.theta[nn.depth]*nn.a[nn.depth-1].T
    nn.sb[nn.depth]=nn.theta[nn.depth]
    #for i in range(nn.depth-1,0,-1):
    for m in range(nn.depth-1,0,-1):
        nn.sw[m]=np.mat(np.ones((nn.w[m].shape[0],nn.w[m].shape[1])))
        for i in range(0,nn.w[m].shape[0]):
            nn.theta[m]=np.matrix([np.array(nn.theta[m+1].T*nn.w[m+1].T[i].T)[0][0]*np.array(nn.a[m][i]*(1-nn.a[m][i]))[0][0] for i in range(0,len(nn.a[m]))]).T
            nn.sw[m]=nn.theta[m]*nn.a[m-1].T
            nn.sb[m]=nn.theta[m]
    return nn
def nn_applygradient(nn):
    for m in range(1,nn.depth+1):
        nn.w[m]-=nn.sw[m]*nn.learning_rate
        nn.b[m]-=nn.sb[m]*nn.learning_rate
    return nn
def annTrain(nn,trainX,trainY):
    for i in range(0,len(trainX)):
        nn = forward(nn,trainX[i])
        nn = nn_backpropagation(nn,trainY[i])
        nn = nn_applygradient(nn)
    return nn
def predict(nn,x):
    forward(nn, x)
    return nn.a[nn.depth]
def test():
    nn = ANN(layer=[1,2,2,2])
    tarinX = [[100],[200],[100],[-100],[200],[-300],[500],[-100]]
    trainY = [[1,0],[1,0],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]]
    nn = annTrain(nn,tarinX,trainY)
    tarinX = [[100],[-200],[500],[-1000]]
    for i in tarinX:
        predict(nn,i)
test()