import tensorflow as tf
import numpy as np

class PINN(object):
    def __init__(self, layers, optimizer, logger, X_f, lb, ub):
        self.u_model = tf.keras.Sequential()
        self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        
        self.u_model.add(tf.keras.layers.Lambda(
            lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
        
        for width in layers[1:-1]:
            self.u_model.add(tf.keras.layers.Dense(
                width, 
                activation=tf.nn.tanh,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                bias_initializer='zeros'
            ))
        
        self.u_model.add(tf.keras.layers.Dense(
            layers[-1], 
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotNormal()
        ))
        
        self.optimizer = optimizer
        self.logger = logger
        self.dtype = tf.float32
        
        self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=self.dtype)
        self.t_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=self.dtype)
        self.X_f = tf.convert_to_tensor(X_f, dtype=self.dtype)
    
    @tf.function
    def loss_fn(self, X_u, u, loss_weights=None):
        if loss_weights is None:
            loss_weights = [1.0, 1.0]
            
        u_pred = self.u_model(X_u)
        data_loss = tf.reduce_mean(tf.square(u - u_pred))
        pde_loss = tf.reduce_mean(tf.square(self.f_model()))
        
        total_loss = loss_weights[0] * data_loss + loss_weights[1] * pde_loss
        return total_loss, data_loss, pde_loss
    
    def __wrap_training_variables(self):
        return self.u_model.trainable_variables

    @tf.function
    def train_step(self, X_u, u, loss_weights=None):
        with tf.GradientTape() as tape:
            total_loss, data_loss, pde_loss = self.loss_fn(X_u, u, loss_weights)
        
        grads = tape.gradient(total_loss, self.__wrap_training_variables())
        self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
        
        return total_loss, data_loss, pde_loss
    
    @tf.function
    def f_model(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_f)
            tape.watch(self.t_f)

            X_f = tf.stack([self.x_f[:,0], self.t_f[:,0]], axis=1)
            u = self.u_model(X_f)
            
            Swc = 0.0
            Sor = 0.0
            M = 2.0
            
            numerator = tf.square(u)
            denominator = numerator + tf.square(1.0 - u) / M
            frac_org = numerator / (denominator + 1e-8)
            
            Sf = tf.sqrt((1.0/M) / (1.0/M + 1.0))
            numerator_Sf = tf.square(Sf - Swc)
            denominator_Sf = numerator_Sf + tf.square(1.0 - Sf - Sor) / M
            frac_Sf = numerator_Sf / (denominator_Sf + 1e-8)
            
            linear_part = (frac_Sf / Sf) * u
            f_u = tf.where(u < Sf, linear_part, frac_org)
    
        u_t = tape.gradient(u, self.t_f)
        u_x = tape.gradient(u, self.x_f)
        f_u_x = tape.gradient(f_u, self.x_f)
        
        del tape
        return u_t + f_u_x
    
    def fit(self, X_u, u, tf_epochs, loss_weights=None):
        self.logger.log_train_start(self)
        
        X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
        u = tf.convert_to_tensor(u, dtype=self.dtype)
        
        self.logger.log_train_opt("Adam")
        
        for epoch in range(tf_epochs):
            total_loss, data_loss, pde_loss = self.train_step(X_u, u, loss_weights)
            
            if epoch % self.logger.frequency == 0:
                custom_info = f"data_loss: {data_loss:.2e}, pde_loss: {pde_loss:.2e}"
                self.logger.log_train_epoch(epoch, total_loss, custom_info)
    
    def predict(self, X):
        return self.u_model(X)