from keras import activations
from keras import backend as K
from keras.engine.topology import Layer

#定义激活函数
def squash(x,axis=1):
    s_squared_norm = K.sum(K.square(x),axis,keepdims=True)+K.epsilon
    scale = K.sqrt(s_squared_norm)/(0.5 + s_squared_norm)
    return scale * x

#定义softmax函数
def softmax(x,axis=-1):
    ex = K.exp(x - K.max(x,axis=axis,keepdims=True))
    return ex/K.sum(ex,axis=axis,keepdims=True)

class Capsule(Layer):
    def _init_(self,num_capsule,dim_capsule,routings=3,share_weights=True,activation='squash',**kwargs):
        super(Capsule,self)._init_(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim-capsule
        self.routing = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation=activation
        else:
            self.activation = activation.get(activation)


    def build(self,input_shape):
        super(Capsule,self).build(input_shape)
        imput_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1,input_dim_capsule,
                                            self.num_capsule * self.dim-capsule),
                                            initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=true)
    def call(self,u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs,self.W,[1],[1])

        else:
            u_hat_vecs = K.conv1d(u_vecs,self.W,[1],[1])


        batch_size = K.shape(u_vecs,self.W)
        input_num_capsule = K.shape(u_vecs)[1]

        u_hat_vecs = K.reshape(u_hat_vecs,(batch_size,input_num_capsule,
                                           self.num_capsule,self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs,[2,2])


        b = K.zeros_like(u_hat_vecs[:,:,:,0])

        for i in range(self.routings):
            c = softmax(b,1)
            o = K.batch_dot(c,u_hat_vecs,[2,2])
            if K.backend() == 'theano':
                b = K.sum(b,axis=1)

        return self.activation(o)


    def compute_output_shape(self,input_shape):
        return(None,self.num_capsule,self.dim_capsule)

            
            
    
