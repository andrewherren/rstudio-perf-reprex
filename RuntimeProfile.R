# Load libraries
library(stochtree)
library(rpart)
library(caret)
library(xgboost)
library(rnn)

# Data generating process specs
n0 <- 50       # of training samples for each coefficient
p <- 10        # of features
n <- n0*(2^p)  # of observations
k <- 2
p1 <- 20       # of nonzero coeff for sparsity
noise <- 0.1

# Generate covariates
xtemp <- as.data.frame(as.factor(rep(0:(2^p-1),n0)))
xtemp1 <- rep(0:(2^p-1),n0)
x <- t(sapply(xtemp1,function(j) as.numeric(int2bin(j,p))))#x <- t(sapply(0:(2^p-1), function(j) as.numeric(int2bin(j,p))))
x <- x*abs(rnorm(length(x))) - (1-x)*abs(rnorm(length(x))) 

# Generate outcome
M <- model.matrix(~.-1,data = xtemp)
M <- cbind(rep(1,n),M)
beta.true <- -10*abs(rnorm(ncol(M)))
beta.true[1] <- 0.5
non_zero_betas <- c(1,sample(1:ncol(M), p1-1))   
beta.true[-non_zero_betas] <- 0      
Y <- M %*% beta.true + rnorm(n, 0,noise)
y<-as.numeric(Y>0)

# Initialize profiler and start runtime tracker
start_time <- Sys.time()
Rprof()
# Proportion of training set
train_prop <- 0.5
# Generate sample indices for the training set
sample_indices <- sample(1:nrow(x), size = floor(train_prop * nrow(x)))

# Split data into train and test
x_train <- as.matrix(x[sample_indices,])
Y_train <- as.numeric(y[sample_indices])
x_test  <- as.matrix(x[-sample_indices,])
Y_test  <- as.numeric(y[-sample_indices])
xtemp_train = as.data.frame(x[sample_indices,])
xtemp_test = as.data.frame(x[-sample_indices,])

# Define BART sampler settings
num_gfr <- 10
num_burnin <- 2
num_mcmc <- 0
num_samples <- num_gfr + num_burnin + num_mcmc

# Run BART
bart <- stochtree::bart(
    X_train = xtemp_train, y_train = Y_train, X_test = xtemp_test, 
    beta = 0.1, alpha = 1, num_trees = 50, num_gfr = num_gfr, 
    num_burnin = num_burnin, num_mcmc = num_mcmc, min_samples_leaf=1, 
    sample_sigma = F, sample_tau = F, sigma2_init = 0.25
)

# Evaluate performance
ypred <- rowMeans(bart$y_hat_test)
acc.bart <- mean(Y_test == round(ypred))

# Assess total runtime of the procedure
end_time <- Sys.time()
print(paste("runtime:", end_time - start_time))

# Summarize profiler
summaryRprof()
