rm(list=ls())


library(maps)
library(mapproj)
library(geoR)
library(ggplot2)
library(fields)
library(rgdal)
library(maptools)
library(spdep)
library(RColorBrewer)
library(classInt)
library(MBA)
library(fields)
library(dplyr)
library(matlib)

#load data
mydata=read.csv("C://Users/user/Documents/AB_NYC_2019.csv",header = TRUE,sep=",")  #data

#load n*p design matrix
z = mydata[!duplicated(mydata[7:8]), ]
#z7 = (z[,7] - min(z[7]))/(max(z[7])-min(z[7]))
#z8 = (z[,8] - min(z[8]))/(max(z[8])-min(z[8]))
#X=cbind(1,z7,z8) ##load the coordinates of the spatial locations
X=cbind(1,z[7:8])
#estimate the beta without taking into account correlations
X=as.matrix(X)
XpX=t(X)%*%X
invXpX=solve(XpX) #nonsingular
price = as.matrix(log(z[,10]+1))
hist(price)
betahat=(invXpX)%*%t(X)%*%price #Price
betahat
#detrend
ztilde=price-X%*%betahat


#load the coordinates of the spatial locations
coord=z[7:8]
coord1 = as.matrix(coord)
#load the knots of the Gaussian kernel
knots=cover.design(coord,100)  #level3.csv
#measurement error square (defined in the Question)
#epssquare=5.6062

#Compute the n*r spatial basis function matrix
r=dim(knots[1:100,])[1]
source("C://Users/user/Documents/Create_GK.R")
source("C://Users/user/Documents/FRKmisslocs2.R")
source("C://Users/user/Documents/EM3.R")
S=Create_GK(coord,cbind(knots[,2],knots[,1]),15,1)
x.res=100
y.res=100
#surf=mba.surf(cbind(coord,S[1]),no.X = x.res,no.Y = y.res,h=5,m=2,extend = FALSE)$xyz.est
#image.plot(surf,xaxs="r",yaxs="r",xlab="lat",ylab="long")
out.EM=EM3(S,ztilde,0,4)
K=matrix(sapply(out.EM[1],as.numeric),r,r)
sigtemp=as.vector(out.EM[2])
sigtemp=sigtemp[[1]]
sigxi=as.numeric(sigtemp[length(sigtemp)])

######Kriging Predictor######                                    
Ksym=(K+t(K))/2 #truncated KL expansion, positive definite
Sp=Create_GK(t(as.matrix(c(40.90,-74.22),1,2)),cbind(knots[,2],knots[,1]),30,1)

out.FRK=FRKmisslocs2(ztilde,S,Sp,Ksym,sigxi,0) #S=G(basis function) in our note, Ksym =K in our note. 
Yhattilde=matrix(sapply(out.FRK[1][1],as.numeric),1,1)
MSPE=matrix(sapply(out.FRK[2][1],as.numeric),1,1)

Yhat=t(as.matrix(c(1,40.90,-74.22)))%*%betahat+Yhattilde #our rogers lat, lon

MSPE #mean squared prediction errors
Yhat #predicted value
#' ###(b) 2. The predicted value and mean squared prediction errors when incorporating the measurement error variance
#compute the EM estimates
out.EM=EM3(S,ztilde,epssquare,4) #changed
K=matrix(sapply(out.EM[1],as.numeric),r,r) #sapply: apply a function over a vector
sigtemp=as.vector(out.EM[2])
sigtemp=sigtemp[[1]]
sigxi=as.numeric(sigtemp[length(sigtemp)])

######Kriging Predictor######
Ksym=(K+t(K))/2
Sp=Create_GK(t(as.matrix(c(30.444,-84.299),1,2)),cbind(knots[,2],knots[,1]),30,1)

out.FRK=FRKmisslocs2(ztilde,S,Sp,Ksym,sigxi,epssquare) #changed
Yhattilde=matrix(sapply(out.FRK[1][1],as.numeric),1,1)
MSPE=matrix(sapply(out.FRK[2][1],as.numeric),1,1)

Yhat=t(as.matrix(c(1,30.444,-84.299)))%*%betahat+Yhattilde 

MSPE #mean squared prediction errors
Yhat #predicted value

#'We can know that when we incorporate the sigma^2 the MSPE is smaller than when we do not incorporate  it. And the predicted value is similar to each other. Hence, we can conclude that the result with incorporating sigma^2 is much better.


#' ### (c) Use the connection between the KL expansion and spatial basis function expansions in class to estimate the eigenvalues associated with the latent process modeled in 3 (b). Plot these eigenvalues. Do these eigenvalues suggest that it is reasonable to truncate the KL expansion? For what practial reason would we like to truncate the KL expansion?
ev=eigen(Ksym) #S%*%K%*%t(S)

plot(ev$values)
title(main = "Plot for Eigenvalues")