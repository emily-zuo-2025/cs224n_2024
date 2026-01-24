class Network(object):
    def __init__(self):
        pass
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmod(np.dot(w, a) + b)
        return a 
    
    
# continue the video -- https://www.youtube.com/watch?v=eMlx5fFNoYc

