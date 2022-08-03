####PART 1 - SUPERVISED LEARNING
########################CLEAR GLOBAL ENVIRONMENT AND LOAD CLASS_DATA###############################
library(ISLR)
library(MASS)
library(boot)
library(glmnet)
library(class)
library(e1071)


hist(y)



###Method 1 - Variable Selection by PCA

#PCA for dimensionality reduction
class_data_o=data.frame(x)
apply(class_data_o,2,mean)
apply(class_data_o,2,var)
max(apply(class_data_o,2,var))
min(apply(class_data_o,2,var))

pca.out=prcomp(class_data_o,scale=TRUE)
pca.out
dim(pca.out$x)
biplot(pca.out,scale=0)
pca.out$sdev
pr.var=pca.out$sdev^2
pr.var
pve=pr.var/sum(pr.var)
pve
plot(pve,xlab="Principal Component",ylab="Proportion of Variance Explained (PVE)",ylim=c(0,1),type='b')
plot(cumsum(pve),xlab="Principal Component",ylab="Cummulative Proportion of Variance Explained (PVE)",ylim=c(0,1),type='b')
xpca=pca.out$x[,1:200]
class_data_pca=data.frame(x=xpca,y=y)

#PCA-Logistic Regression with CV
K=10
set.seed(10)
n = nrow(xpca) 
folds = sample(1:K, n, replace=TRUE) 
fold.error.pca = rep(0, K) 
for (j in 1:K)
{
  train=folds!=j
  glm.fit2=glm(y~.,data=class_data_pca,family=binomial,subset=train)
  glm.probs2=predict(glm.fit2,newdata = class_data_pca[-train,],type="response")
  glm.pred2=ifelse(glm.probs2>0.5,1,0)
  fold.error.pca[j]=mean(glm.pred2!=class_data_pca$y[-train])
}
CV.error.pca = sum(fold.error.pca)/K
fold.error.pca
CV.error.pca

##PCA- Linear Discriminant Analysis with CV
K=10
set.seed(10)
n = nrow(xpca) 
folds = sample(1:K, n, replace=TRUE) 
fold.error.plda = rep(0, K) 
for (j in 1:K)
{
  train=folds!=j
  lda.fit2=lda(y~.,data=class_data_pca,subset=train)
  lda.pred2=predict(lda.fit2,class_data_pca[-train,])
  fold.error.plda[j]=mean(lda.pred2$class!=class_data_pca$y[-train])
}
CV.error.plda = sum(fold.error.plda)/K
fold.error.plda
CV.error.plda

## PCA - K-Nearest Neighbors with CV
library(class)
K=10
set.seed(10)
n = nrow(xpca) 
folds = sample(1:K, n, replace=TRUE) 
fold.error.pknn = rep(0, K) 
for (j in 1:K)
{
  train=folds!=j
  knn.pred2=knn(xpca[train,],xpca[-train,],class_data_pca$y[train],k=3)
  table(knn.pred2,class_data_pca$y[-train])
  fold.error.pknn[j]=mean(knn.pred2!=class_data_pca$y[-train])
}
CV.error.pknn = sum(fold.error.pknn)/K
fold.error.pknn
CV.error.pknn

##Regularization
set.seed(10)
xr=xpca
yr=y
xrx=as.matrix(xr)
yrx=as.matrix(yr)
xrx=scale(xrx)
cv.lasso=cv.glmnet(xrx,yrx,family=binomial,alpha=0)
plot(cv.lasso)
plot(log(cv.lasso$lambda),cv.lasso$cvm)
lambda=cv.lasso$lambda.1se
cv.lasso
lambda

#Regularization - Refitting Logistic Regression with Ridge/Lasso and CV with 1se lambda
K=10
set.seed(10)
n = nrow(xr) 
folds = sample(1:K, n, replace=TRUE) 
fold.error.final = rep(0, K) 
for (j in 1:K)
{
  
  xrx=as.matrix(xr)
  yrx=as.matrix(yr)
  xrx=scale(xrx)
  lasso.fit=glmnet(xrx[folds!=j,],yrx[folds!=j],family=binomial,alpha=0,lambda=cv.lasso$lambda.1se)#cv.lasso$lambda.1se)
  lasso.probs=predict(lasso.fit,newx=xrx[folds==j,],s=cv.lasso$lambda.1se,type="response")
  lasso.pred=ifelse(lasso.probs>0.5,1,0)
  fold.error.final[j]=mean(lasso.pred!=yrx[folds==j])
}
CV.error.final = sum(fold.error.final)/K
fold.error.final
CV.error.final



###Method 2 - Variable Selection by Correlation Test
#Clear all memory and reload x,xnew and y

#Correlation test for variables in X
corr_x=cor(x)
xx=x
max(corr_x)
min(corr_x)
mean(corr_x)


for(i in 1:ncol(corr_x))
{
  for(j in i:nrow(corr_x))
  {
    if(i==j){next}
    if(abs(corr_x[j,i])>0.60)
      {
       
        drop=c(paste("V",j,sep=""))
        xx=xx[,!(names(xx)%in%drop)]
        
      }
  }
}
min(cor(xx))
n=nrow(xx)
p=ncol(xx)
yy=data.frame(y)


#Logistic Regression with CV(with variable selection)
K=10
set.seed(10)
n = nrow(xx) 
folds = sample(1:K, n, replace=TRUE) 
fold.error = rep(0, K) 
for (j in 1:K)
{
  train=folds!=j
  xt=xx[train,]
  yt=yy[train,]
  c=apply(xt,2,function(a) cor(a,yt))
  ind = order(abs(c))[(p-99):p]
  xr=xx[,ind]
  yr=yy
  class_data = data.frame(x=xr,y=yr)
  
  glm.fit=glm(y~.,data=class_data,family=binomial,subset=train)
  glm.probs=predict(glm.fit,newdata = class_data[-train,],type="response")
  glm.pred=ifelse(glm.probs>0.5,1,0)
  fold.error[j]=mean(glm.pred!=class_data$y[-train])
}
CV.error = sum(fold.error)/K
fold.error
CV.error



## Linear Discriminant Analysis with CV(with variable selection)
K=10
set.seed(10)
n = nrow(xx) 
folds = sample(1:K, n, replace=TRUE) 
fold.error.lda = rep(0, K) 
for (j in 1:K)
{
  train=folds!=j
  xt=xx[train,]
  yt=yy[train,]
  c=apply(xt,2,function(a) cor(a,yt))
  ind = order(abs(c))[(p-99):p]
  xr=xx[,ind]
  yr=yy
  class_data = data.frame(x=xr,y=yr)
  
  lda.fit=lda(y~.,data=class_data,subset=train)
  lda.pred=predict(lda.fit,class_data[-train,])
  fold.error.lda[j]=mean(lda.pred$class!=class_data$y[-train])
}
CV.error.lda = sum(fold.error.lda)/K
fold.error.lda
CV.error.lda


## K-Nearest Neighbors with CV(with variable selection)
library(class)
K=10
set.seed(10)
n = nrow(xx) 
folds = sample(1:K, n, replace=TRUE) 
fold.error.knn = rep(0, K) 
for (j in 1:K)
{
  train=folds!=j
  xt=xx[train,]
  yt=yy[train,]
  c=apply(xt,2,function(a) cor(a,yt))
  ind = order(abs(c))[(p-99):p]
  xr=xx[,ind]
  yr=yy
  class_data = data.frame(x=xr,y=yr)
  
  knn.pred=knn(xr[train,],xr[-train,],class_data$y[train],k=3)
  table(knn.pred,class_data$y[-train])
  fold.error.knn[j]=mean(knn.pred!=class_data$y[-train])
}
CV.error.knn = sum(fold.error.knn)/K
fold.error.knn
CV.error.knn


###Regularization
set.seed(11)
xrx=as.matrix(xr)
yrx=as.matrix(yr)
xrx=scale(xrx)
cv.lasso=cv.glmnet(xrx,yrx,family=binomial,alpha=0,nfolds=10)
plot(cv.lasso)
plot(log(cv.lasso$lambda),cv.lasso$cvm)
lambda=cv.lasso$lambda.1se
cv.lasso
lambda
#Regularization - Refitting Logistic Regression with Ridge/Lasso and CV with 1se lambda
K=10
set.seed(11)
n = nrow(xx) 
folds = sample(1:K, n, replace=TRUE) 
fold.error.final = rep(0, K) 
for (j in 1:K)
{

  xrx=as.matrix(xr)
  yrx=as.matrix(yr)
  xrx=scale(xrx)
  lasso.fit=glmnet(xrx[folds!=j,],yrx[folds!=j],family=binomial,alpha=0,lambda=cv.lasso$lambda.1se)#cv.lasso$lambda.1se)
  lasso.probs=predict(lasso.fit,newx=xrx[folds==j,],s=cv.lasso$lambda.1se,type="response")
  lasso.pred=ifelse(lasso.probs>0.5,1,0)
  fold.error.final[j]=mean(lasso.pred!=yrx[folds==j])
}
CV.error.final = sum(fold.error.final)/K
fold.error.final
CV.error.final


## Making predictions on xnew
xtestr=xnew[,ind]
xtest=as.matrix(xtestr)
xtest=scale(xtest)
final.probs=predict(lasso.fit,newx=xtest,s=cv.lasso$lambda.1se,type="response")
final.pred=ifelse(final.probs>0.5,1,0)
ynew=final.pred
test_error=CV.error.final
save(ynew,test_error,file="37.RData")

###PART 2 - UNSUPERVISED LEARNING
########################CLEAR GLOBAL ENVIRONMENT AND LOAD CLUSTER_DATA###############################
library(cluster)
clus_data_o=data.frame(y)
dim(clus_data_o)
row.names(clus_data_o)
names(clus_data_o)
apply(clus_data_o, 2, mean) 
apply(clus_data_o, 2, var)
max(apply(clus_data_o, 2, var))
min(apply(clus_data_o, 2, var))
pca.out=prcomp(clus_data_o, scale=TRUE)
pca.out
names(pca.out)
pca.out$sdev
pca.out$center
pca.out$scale
pca.out$rotation
pca.out$x
dim(pca.out$x)
apply(pca.out$x, 2, sd)
biplot(pca.out, scale=0)
pca.out$rotation=-pca.out$rotation
pca.out$x=-pca.out$x
biplot(pca.out, scale=0)
pca.out$sdev
pr.var=pca.out$sdev^2
pr.var
pve=pr.var/sum(pr.var)
pve
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(pca.out$x[,1:100])


clus_data=data.frame(pca.out$x[,1:10])

#K-means clustering
km.out=kmeans(clus_data,centers=5,nstart=15)
km.out
plot(clus_data,col=km.out$cluster,cex=2,pch=1,lwd=2)


km.out=kmeans(clus_data,centers=10,nstart=15)
km.out
plot(clus_data,col=km.out$cluster,cex=2,pch=1,lwd=2)

clus_data=data.frame(pca.out$x[,1:2])
km.out=kmeans(clus_data,centers=5,nstart=15)
km.out
plot(clus_data,col=km.out$cluster,cex=2,pch=1,lwd=2)

km.out=kmeans(clus_data,centers=8,nstart=15)
km.out
plot(clus_data,col=km.out$cluster,cex=2,pch=1,lwd=2)

#DBscan
library(dbscan)
db = dbscan(clus_data, eps = 7, minPts = 5) 
plot(clus_data,col=db$cluster+1,cex=2,pch=1,lwd=2)

#Hierarchical Clustering
clus_data=data.frame(pca.out$x[,1:10])
hc.complete=hclust(dist(clus_data),method="complete")
plot(hc.complete)
hc.cut=cutree(hc.complete,33)

#Gaussian mixture model (GMM)
library(mclust)
gmm = Mclust(clus_data, G = 4)
plot(clus_data,col=gmm$classification,cex=2,pch=1,lwd=2)

#Cluster-wsie Dissimilarity (euclidean distance) with K means
clus_data=data.frame(pca.out$x[,1:100])
c=50
dissim=rep(0,c)
for(i in 1:c)
{
  km.out=kmeans(clus_data,centers=i,nstart=15)
  for(j in 1:i)
  {
    dissim[i]= sum(dissim[i])+sum(daisy(clus_data[km.out$cluster==j,],metric="euclidean"))
  }
}
plot(dissim)

#Reduction in dissimilarity is very less after cluster size = 10, which serves as the elbow point. 
#Thus optimum number of clusters =10 by the Dissimilarity matrix approach.

#Reducing size of data so as to obtain a plot and avoid "figure margins too large error"
clus_data=data.frame(pca.out$x[,1:5])
km.out=kmeans(clus_data,centers=10,nstart=15)
plot(clus_data,col=km.out$cluster,cex=2,pch=1,lwd=2)
