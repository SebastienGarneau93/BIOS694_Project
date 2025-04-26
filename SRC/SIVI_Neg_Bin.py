#We import some libraries
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines

#We define some log pdf used

#Variational distribution for r
def lognormal(z,mu,sigma):
    log_pdf = -tf.math.log(z)-tf.math.log(sigma)-0.5*tf.math.log(2*pi)-0.5*tf.math.square(tf.math.log(z)-mu)/tf.math.square(sigma)
    return log_pdf

#Variational distribution for p
def logitnomarl(z,mu,sigma):
    logit = tf.math.log(z/(1-z))
    log_pdf = -tf.math.log(sigma)-0.5*tf.math.log(2*pi)-0.5*tf.math.square(logit-mu)/tf.math.square(sigma)-tf.math.log(z*(1-z))
    return log_pdf

#How to sample from lognormal
def sample_ln(mu, sigma):
    eps = tf.random.normal(shape=tf.shape(mu))
    z = tf.exp(mu + eps * sigma)
    return z

#how to sample from logitnormal
def sample_logitn(mu, sigma):
    eps = tf.random.normal(shape=tf.shape(mu))
    z = mu + eps * sigma
    res = tf.exp(z)/(1+tf.exp(z))
    return res

#We define q(psi)
def sample_hyper(J,noise_dim,neuralnet,z_dim):
    #We generate the random error used as input
    error = tf.random.normal(shape=(J,noise_dim))
    #We obtain the outcome from neural network
    hyper = neuralnet(error)
    #We reshape the output, not useful in this case, but will be for more complicated model
    hyper = tf.reshape(hyper, [-1,z_dim])
    return hyper

def Compute_z_sample_k(z_sample):
    z_sample_0 = tf.expand_dims(z_sample,axis=0)
    z_sample_k = tf.tile(z_sample_0,[K+1,1,1]) 
    return z_sample_k

def sample_psi_star(K,noise_dim,neuralnet,z_dim,psi_sample):
    psi_star_0 = sample_hyper(K,noise_dim,neuralnet,z_dim)
    psi_star_1 = tf.expand_dims(psi_star_0,axis=1)
    #We copy them among the second axis, such that for each value from the J sample, we have the whole set of K values as well
    psi_star_2 = tf.tile(psi_star_1,[1,J,1])
    #We add them to the J sample. Has shape (K+1,J,z_dim)
    psi_star = tf.concat([psi_star_2, tf.expand_dims(psi_sample,axis=0)],0)
    return psi_star

def Compute_log_H(r_sample_k,mu_star_r,sigma_r,p_sample_k,mu_star_p,sigma_p):
    term1 = lognormal(r_sample_k,mu_star_r,sigma_r)
    term2 = logitnomarl(p_sample_k,mu_star_p,sigma_p)
    sum_term = term1 + term2
    log_H = tfp.math.reduce_logmeanexp(sum_term,axis=0)
    return log_H

def Compute_log_P(Y,r_sample,p_sample,a,b,c,d):
    log_likelihood = tf.reduce_sum(tf.math.lgamma(tf.transpose(Y)+r_sample),axis=1)-N*tf.math.lgamma(r_sample)-tf.reduce_sum(tf.math.lgamma(Y+1),axis=0) + tf.reduce_sum(Y)*tf.math.log(p_sample)+N*r_sample*tf.math.log(1-p_sample)
    log_prior = (a-1)*tf.math.log(r_sample)-b/r_sample+a*tf.math.log(b)-tf.math.lgamma(a) + (c-1)*tf.math.log(p_sample) + (d-1)*tf.math.log(1-p_sample)+tf.math.lgamma(c+d)-tf.math.lgamma(c)-tf.math.lgamma(d)
    log_P = log_likelihood+log_prior
    return log_P

def Compute_loss(Y,r_sample,p_sample,a,b,c,d,r_sample_k,mu_star_r,sigma_r,p_sample_k,mu_star_p,sigma_p):
    y_pred = Compute_log_H(r_sample_k,mu_star_r,sigma_r,p_sample_k,mu_star_p,sigma_p)
    y_true = Compute_log_P(Y,r_sample,p_sample,a,b,c,d)
    loss = tf.reduce_mean(y_pred - y_true)
    return loss

#We define some constants
pi = tf.constant(np.pi)
noise_dim = 10
K=1000
J=50
N=150
a=b=c=d=0.01 #prior parameters for r and p
sigma_r = sigma_p = tf.constant(0.1)
z_dim = 2
#We copy the data
Y = np.concatenate([np.zeros(70),np.ones(38),2*np.ones(17),3*np.ones(10),4*np.ones(9),
              5*np.ones(3),6*np.ones(2),7*np.ones(1)])
Y_tensor = tf.constant(Y,shape=(150,1),dtype=tf.float32)

#We define the neural network
neuralnet = tf.keras.Sequential()
neuralnet.add(tf.keras.layers.Input(shape=(noise_dim,)))
neuralnet.add(tf.keras.layers.Dense(30, activation="relu"))
neuralnet.add(tf.keras.layers.Dense(60, activation="relu"))
neuralnet.add(tf.keras.layers.Dense(30, activation="relu"))
neuralnet.add(tf.keras.layers.Dense(2, activation=None))

#We optimize the model
#We pick the optimizer 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

#We generate our psi and z samples
epochs = 2000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        psi_sample = sample_hyper(J,noise_dim,neuralnet,z_dim)
        mu_r = psi_sample[:,0]
        mu_p = psi_sample[:,1]
        r_sample = sample_ln(mu_r,sigma_r)
        r_sample = tf.reshape(r_sample,(J,1))
        p_sample = sample_logitn(mu_p,sigma_p)
        p_sample = tf.reshape(p_sample,(J,1))
        psi_star = sample_psi_star(K,noise_dim,neuralnet,z_dim,psi_sample)
        mu_star_r = psi_star[:,:,0:1]
        mu_star_p = psi_star[:,:,1:2] 
        r_sample_k = Compute_z_sample_k(r_sample)
        p_sample_k = Compute_z_sample_k(p_sample)
        loss_value = Compute_loss(Y_tensor,r_sample,p_sample,a,b,c,d,r_sample_k,mu_star_r,sigma_r,p_sample_k,mu_star_p,sigma_p)
    
    #We retrieve the gradients
    grads = tape.gradient(loss_value, neuralnet.trainable_variables)
    #We apply them on the variables such that they minimize the loss
    optimizer.apply_gradients(zip(grads, neuralnet.trainable_variables))
    
    #We print the loss
    print("Loss values is:", loss_value)

#Mean-field results as given by the author
mf_ga_a = 1.148120480673975e+02
mf_ga_b = 1.149553294523141e+02 
mf_be_a = 1.720000010000000e+02
mf_be_b = 1.498130396138695e+02
r_mf = stats.gamma.rvs(mf_ga_a,scale=1/mf_ga_b,size=2000)
p_mf = stats.beta.rvs(mf_be_a, mf_be_b, size=2000)

#Load the results from MCMC
MCMC_sample =  pd.read_csv("Post_sample_MCMC.csv")
MCMC_p = MCMC_sample["p"]
MCMC_r = MCMC_sample["r"]

#Posterior sample
mu_r_hive = []
mu_p_hive = []
p_sivi = []
r_sivi = []
for i in range(500):
    error2 = tf.random.normal(shape=(K,noise_dim))
    psi_sample = sample_hyper(J,noise_dim,neuralnet,z_dim)
    mu_r = psi_sample[:,0]
    mu_p = psi_sample[:,1]
    r_sample_hive = sample_ln(mu_r,sigma_r)
    p_sample_hive = sample_logitn(mu_p,sigma_p)
    mu_r_hive.extend(np.squeeze(mu_r))
    mu_p_hive.extend(np.squeeze(mu_p))
    p_sivi.extend(np.squeeze(p_sample_hive))
    r_sivi.extend(np.squeeze(r_sample_hive))

#We plot the implicit distribution of mu_r and mu_p
gg = sns.kdeplot(x=np.array(mu_r_hive),y=np.array(mu_p_hive),n_levels=10)
plt.xlim([-0.65,0.75])
#We save the plot
plt.savefig("Implicit.png",dpi = 300, bbox_inches='tight')

#For p
sns.kdeplot(p_sivi, label = "SIVI")
sns.kdeplot(p_mf, label = "Mean-Field VI")
sns.kdeplot(MCMC_p,label = "MCMC")
plt.legend()
plt.savefig("p_hist.png",dpi = 300, bbox_inches='tight')

#For r
sns.kdeplot(r_sivi, label = "SIVI")
sns.kdeplot(r_mf, label = "Mean-Field VI")
sns.kdeplot(MCMC_r,label = "MCMC")
plt.legend()
plt.savefig("r_hist.png",dpi = 300, bbox_inches='tight')

#Joint distribution
fig,ax = plt.subplots()
sns.kdeplot(x=r_sivi,y=p_sivi,ax=ax,color = "blue", fill=False)
sns.kdeplot(x=r_mf,y=p_mf,ax=ax, color = "orange",fill=False)
sns.kdeplot(x=MCMC_r,y=MCMC_p,ax=ax, color = "green",fill=False)

#Manually create the legend
handles = [
    mlines.Line2D([], [], color='blue', label='SIVI'),
    mlines.Line2D([], [], color='orange', label='Mean-Field VI'),
    mlines.Line2D([], [], color='green', label='MCMC')
]

# Add the custom handles to the legend
ax.legend(handles=handles)
plt.savefig("Joint.png",dpi = 300, bbox_inches='tight')
