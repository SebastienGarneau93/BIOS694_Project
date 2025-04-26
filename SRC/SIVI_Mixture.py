#We import some libraries
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

#We define some log pdf used

#Normal distribution
def normal(z,mu,sigma):
    log_pdf = -0.5*tf.math.square(z-mu)/tf.math.square(sigma)-tf.math.log(sigma)-0.5*tf.math.log(2*pi)
    return log_pdf

#We define q(psi)
def sample_hyper(J,noise_dim,neuralnet,z_dim):
    #We generate the random error used as input
    error = tf.random.normal(shape=(J,noise_dim))
    #We obtain the outcome from neural network
    hyper = neuralnet(error)
    #We reshape the output, not useful in this case, but will be for more complicated model
    hyper = tf.reshape(hyper, [-1,z_dim])
    return hyper

#We compute psi_star : Psi stars are the K extra psi we sample
def sample_psi_star(K,noise_dim,neuralnet,z_dim,psi_sample):
    psi_star_0 = sample_hyper(K,noise_dim,neuralnet,z_dim)
    psi_star_1 = tf.expand_dims(psi_star_0,axis=1)
    #We copy them among the second axis, such that for each value from the J sample, we have the whole set of K values as well
    psi_star_2 = tf.tile(psi_star_1,[1,J,1])
    #We add them to the J sample. Has shape (K+1,J,z_dim)
    psi_star = tf.concat([psi_star_2, tf.expand_dims(psi_sample,axis=0)],0)
    return psi_star
    

#Let's compute the z_sample_k : We repeat the J z sample such that we have K+1,J,z_dim shape
def Compute_z_sample_k(z_sample):
    z_sample_0 = tf.expand_dims(z_sample,axis=0)
    z_sample_k = tf.tile(z_sample_0,[K+1,1,1]) 
    return z_sample_k

#We define log_H : the variational distribution
def Compute_log_H(psi_star,z_sample_k,sigma):
    term1 = normal(z_sample_k,psi_star,sigma)
    #Compute log(mean(exp(term1)) which is  log[ 1/(K+1) q(z|psi)+sum(q(z|psi_k))]
    log_H = tfp.math.reduce_logmeanexp(term1,axis=0)
    return log_H

#We define log_p : Target distribution
def Compute_log_P(z_sample):
    log_P = tf.math.log((0.3/tf.math.sqrt(2*pi))*tf.exp(-tf.square(z_sample+2)/2)+(0.7/tf.math.sqrt(2*pi))*tf.exp(-tf.square(z_sample-2)/2))
    return log_P

#We define the loss function
def Compute_loss(z_sample,psi_star,z_sample_k,sigma):
    y_pred = Compute_log_H(psi_star,z_sample_k,sigma)
    y_true = Compute_log_P(z_sample)
    loss = tf.reduce_mean(y_pred - y_true)
    return loss

pi = tf.constant(np.pi)
#Number of psi using in the correction
K= 100
#number of MC sample
J = 20
#Dimension of the parameters
z_dim = 1

#SD of the variational distribution
sigma = tf.constant(0.1, dtype=tf.float32)

#We define the model
noise_dim = 10
neuralnet = tf.keras.Sequential()
neuralnet.add(tf.keras.layers.Input(shape=(noise_dim,)))
neuralnet.add(tf.keras.layers.Dense(30, activation="relu"))
neuralnet.add(tf.keras.layers.Dense(60, activation="relu"))
neuralnet.add(tf.keras.layers.Dense(30, activation="relu"))
neuralnet.add(tf.keras.layers.Dense(1, activation=None))

#We optimize the model
epochs = 5000

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        psi_sample = sample_hyper(J,noise_dim,neuralnet,z_dim)
        z_sample = tf.random.normal(mean=psi_sample,stddev=sigma,shape=tf.shape(psi_sample))
        psi_star = sample_psi_star(K,noise_dim,neuralnet,z_dim,psi_sample)
        z_sample_k = Compute_z_sample_k(z_sample)
        loss_value = Compute_loss(z_sample,psi_star,z_sample_k,sigma)
    
    #We retrieve the gradients
    grads = tape.gradient(loss_value, neuralnet.trainable_variables)
    #We apply them on the variables such that they minimize the loss
    optimizer.apply_gradients(zip(grads, neuralnet.trainable_variables))
    
    #We print the loss
    print("Loss values is:", loss_value)

#We generate a sample using SIVI
z_sivi = []
#Each gives 20
for i in range(500):
    psi_sample = sample_hyper(J,noise_dim,neuralnet,z_dim)
    z_hive = tf.random.normal(mean=psi_sample,stddev=sigma,shape=tf.shape(psi_sample))
    z_sivi.extend(np.squeeze(z_hive))

#We compute the true pdf

yy=[]
xx = np.arange(-10,10,0.01)
for r in xx:
        pdf = 0.3*stats.norm.pdf(r, loc=-2, scale=1)+0.7*stats.norm.pdf(r, loc=2, scale=1) #gmm
        yy.append(pdf)

#We plot both
g = sns.displot(z_sivi,kind="kde")
for ax in g.axes.flat:
    line = ax.lines[0]
    line.set_label("SIVI approximation")
    ax.plot(xx,yy,"y-",label= "Target pdf")
    ax.legend(loc="best")

#We save the plot
plt.savefig("Mixture.png",dpi = 300, bbox_inches='tight')
