### BIOS 694 PROJECT : MCMC for neg.Bin. example
library(nimble)
#We recreate the data
Y = c(rep(0,70),rep(1,38),rep(2,17),rep(3,10),rep(4,9),rep(5,3),rep(6,2),7)

#prior parameters
a = 0.01
b = a
c = b
d = c

Nimble.code <- nimbleCode({
  #priors
  p ~ dbeta(c,d)
  r ~ dgamma(shape = a, rate = b)
  
  for (i in 1:n) {
    #Likelihood
    #Use 1-p to match the parameterization of SIVI
    y[i] ~ dnegbin(1-p,r)
  }
})

#The constants in the code
Nimble.const <- list(a=a,b=b,c=c,d=d,n=length(Y))
#The data
Nimble.data <- list(y=Y)

#Parameters of interests
Nimble.param <- list("p","r")


#Starting values
Nimble.ini <- list(p = 0.5, r = 1)

#Compile the model to verify that everything's alright
Nimble.model <- nimbleModel(code = Nimble.code, name = "NegBin", constants = Nimble.const,
                             data = Nimble.data, inits = Nimble.ini)

#Run the MCMC
MCMC <- nimbleMCMC(
  code = Nimble.code,
  data = Nimble.data,
  constants = Nimble.const,
  inits = Nimble.ini,
  monitors = Nimble.param,
  niter = 10000,
  nburnin = 5000,
  nchains = 4,
  summary = TRUE,
  samplesAsCodaMCMC = T
)

#We extract the posterior sample
Posterior_samples <-as.matrix(MCMC$samples)

#We save it in a CSV file for python

write.csv(Posterior_samples,file = "Post_sample_MCMC.csv",row.names = F)
