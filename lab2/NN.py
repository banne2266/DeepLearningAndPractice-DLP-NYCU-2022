import numpy as np

class optimizer():
    def __init__(self, name = 'SGD', lr = 0.001, hyper_parameter = {}):
        self.name = name
        self.lr = lr
        self.hyper_parameter = hyper_parameter


class FC_layer():
    def __init__(self, input_size=16, output_size=16, bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias

        self.weight_matrix = np.random.uniform(-1, 1, (input_size, output_size))
        if self.bias:
            self.bias_matrix = np.random.uniform(-1, 1, (1, output_size))

        self.trained = 0
        
    def forward(self, x:np.array):
        self.inputs = x
        self.batch_size = x.shape[0]
        if self.bias:
            return np.dot(x, self.weight_matrix) + self.bias_matrix
        else:
            return np.dot(x, self.weight_matrix)

    def backward(self, dy:np.array, opti:optimizer):
        dx = np.dot(dy, np.transpose(self.weight_matrix))
        
        lr = opti.lr
        dw = np.dot(np.transpose(self.inputs), dy) / self.batch_size
        if self.bias:
            db = np.sum(dy, axis=0) / self.batch_size
            

        if opti.name == 'momentum':
            beta = opti.hyper_parameter['beta']
            if self.trained ==  0:
                self.velocity = np.zeros((self.input_size, self.output_size))
                if self.bias:
                    self.velocity_b = np.zeros((1, self.output_size))

            self.velocity = beta * self.velocity - lr * dw
            self.weight_matrix = self.weight_matrix + self.velocity
            if self.bias:
                self.velocity_b = beta * self.velocity_b - lr * db
                self.bias_matrix = self.bias_matrix + self.velocity_b

        elif opti.name == 'adagrad':
            epsilon = opti.hyper_parameter['epsilon']
            if self.trained == 0:
                self.v = np.zeros((self.input_size, self.output_size))
                if self.bias:
                    self.v_b = np.zeros((1, self.output_size))
            
            self.v = self.v + (dw ** 2)
            self.weight_matrix = self.weight_matrix - lr * dw / np.sqrt(self.v + epsilon)
            
            if self.bias:
                self.v_b = self.v_b + (db ** 2)
                self.bias_matrix = self.bias_matrix - lr * db / np.sqrt(self.v_b + epsilon)      
            
        elif opti.name == 'adam':
            epsilon = opti.hyper_parameter['epsilon']
            beta1 = opti.hyper_parameter['beta1']
            beta2 = opti.hyper_parameter['beta2']
            if self.trained == 0:
                self.m = np.zeros((self.input_size, self.output_size))
                self.v = np.zeros((self.input_size, self.output_size))
                if self.bias:
                    self.m_b = np.zeros((1, self.output_size))
                    self.v_b = np.zeros((1, self.output_size))
            
            self.m = beta1 * self.m + (1 - beta1) * dw
            self.v = beta2 * self.v + (1 - beta2) * (dw ** 2)
            m_hat = self.m / (1 - beta1)
            v_hat = self.v / (1 - beta2)
            self.weight_matrix = self.weight_matrix - lr * m_hat  / np.sqrt(v_hat + epsilon)
            
            if self.bias:
                self.m_b = beta1 * self.m_b + (1 - beta1) * db
                self.v_b = beta2 * self.v_b + (1 - beta2) * (db ** 2)
                m_b_hat = self.m_b / (1 - beta1)
                v_b_hat = self.v_b / (1 - beta2)
                self.bias_matrix = self.bias_matrix - lr * m_b_hat  / np.sqrt(v_b_hat + epsilon)
        
        else: #SGD
            self.weight_matrix = self.weight_matrix - lr * dw
            if self.bias:
                self.bias_matrix = self.bias_matrix - lr * db

        self.trained = 1
        return dx


class relu():
    def __init__(self):
        pass

    def forward(self, x:np.array):
        self.inputs = x
        return self.relu(x)
    
    def relu(self, x:np.array):
        return np.maximum(x, 0)

    def backward(self, dy:np.array, opti:optimizer):
        dx = np.maximum(self.inputs, 0) * dy
        return dx


class sigmoid():
    def __init__(self):
        pass

    def forward(self, x:np.array):
        self.inputs = x
        return self.sigmoid(x)
    
    def sigmoid(self, x:np.array):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, dy:np.array, opti:optimizer):
        dx = self.sigmoid(self.inputs) * (1 - self.sigmoid(self.inputs)) * dy
        return dx


class myNN():
    def __init__(self, layers=[]):
        self.layers = layers

    def forward(self, x:np.array):
        data = x
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def backward(self, dy:np.array, opti:optimizer):
        self.layers.reverse()
        for layer in self.layers:
            dy = layer.backward(dy, opti)
        self.layers.reverse()


class MSE():
    def __init__(self):
        self.size = 0
    
    def forward(self, y:np.array, y_pred:np.array):
        self.size = y.shape[0]
        loss = (y - y_pred) ** 2
        loss = np.sum(loss) / self.size
        return loss

    def backward(self:np.array, y, y_pred:np.array):
        dy = 2 * (y_pred - y)
        return dy


