##################################################################################################
## Construct a data set with 200 points using the following steps:
##  Generate 10 two-dimensional vectors, m0, , m1, . . . , m9 ∈ R^2, such that they are 
##  i.i.d.~  N ([1,0],[[1,0][0,1]]).  Generate 10 more two-dimensional vectors,
## m0',m1',....,m9'∈ R^2,  such that they are i.i.d. ~  N ([0,1],[[1,0][0,1]]).


library(MASS)
yvalue = function(c,m,x) {
  result = -c-m*x
  return(result)
}

euclidean_dist=function(pt1,pt2,pt3,pt4){
  return (sqrt((pt1-pt2)**2+(pt3-pt4)**2))
}

##################################################################################################
## Generate 10 data points such that they are i.i.d. ~ N (mi,[[0.1,0][0,0.1]])
## for all i = 0, 1, . . . , 9.
## Thus a 100 feature vectors are generated. Label them all as (+1).

## Generate 10 data points such that they are i.i.d. ~ N (mi',[[0.1,0][0,0.1]])
## for all i = 0, 1, . . . , 9.
## Thus a 100 feature vectors are generated. Label them all as (-1).


## Generating the 100 datapoints with given condition 1
m1=c(1,0)
m2=c(0,1)
cov=matrix(c(1,0,0,1),2,2,byrow=TRUE)
d1= mvrnorm(n = 10,mu = m1,Sigma = cov)
vv1=matrix(rep(1,100))
vv2=matrix(rep(-1,100))
data1=mvrnorm(n = 10,mu =c(d1[1,1],d1[1,2]),Sigma = matrix(c(0.1,0,0,0.1),2,2,byrow=TRUE))
head(data1)
for(i in 2:10){
  temp=mvrnorm(n = 10,mu =c(d1[i,1],d1[i,2]),Sigma = matrix(c(0.1,0,0,0.1),2,2,byrow=TRUE))
  data1=rbind(data1,temp)
}

## Assigning (+1) to first dataset
## Adding bias term to each datapoint

data1=cbind(data1,vv1)
data1=cbind(vv1,data1)

ll1=c()
for (i in 1:10){
  temp=mvrnorm(n = 10,mu =c(d1[i,1],d1[i,2]),Sigma = matrix(c(0.1,0,0,0.1),2,2,byrow=TRUE))
  ll1=append(ll1,temp)
}
lll1=matrix(ll1,100,2,byrow=TRUE)



## Generating the 100 datapoints with given condition 2

d2= mvrnorm(n = 10,mu = m2,Sigma = cov)
data2=mvrnorm(n = 10,mu =c(d2[1,1],d2[1,2]),Sigma = matrix(c(0.1,0,0,0.1),2,2,byrow=TRUE))
for(i in 2:10){
  temp=mvrnorm(n = 10,mu =c(d2[i,1],d2[i,2]),Sigma = matrix(c(0.1,0,0,0.1),2,2,byrow=TRUE))
  data2=rbind(data2,temp)
}

## Assigning (+1) to first dataset
## Adding bias term to each datapoint
data2=cbind(data2,vv2)
data2=cbind(vv1,data2)

## A dataset of 200 datapoints is ready to implement model
data=rbind(data1,data2)


##################################################################################################
## Plot the generated features in the form of a scatterplot. Represent the
## features having different labels with different colors.


## This step is just to resize the plot
xmin=min(data[1:200,2])
xmax=max(data[1:200,2])
ymin=min(data[1:200,3])
ymax=max(data[1:200,3])

## Finding the maximuma and minimum value of dataset. so that, we can plot the datapoints perfectly

aa=c(xmin-0.2,xmax+0.2)
bb=c(ymin-0.2,ymax+0.2)
plot(aa,bb)
points(data1[1:100,2],data1[1:100,3],col='red',cex=0.7)
points(data2[1:100,2],data2[1:100,3],col='blue',pch=18,cex=1)


##################################################################################################
## Compute the classifier using the linear model. Plot the classifying line
## along with the scatter plot of the generated features, showing features
## above and below the line. Print the training error (i.e., the average of
## misclassified data) for the classifier computed.


## Estimating the feature vector est_beta
temp_mat=solve(t(data[1:200,1:3])%*%data[1:200,1:3])
est_beta=(temp_mat%*%t(data[1:200,1:3]))%*%data[1:200,4]

x=seq(xmin-0.2,xmax+0.2,length=100)
y=-(est_beta[1,1]/est_beta[3,1])-((est_beta[2,1]/est_beta[3,1])*x)


## plotting aa and bb to resize the plot
plot(aa,bb)
points(data1[1:100,2],data1[1:100,3],col='red',cex=0.7)
points(data2[1:100,2],data2[1:100,3],col='blue',pch=18,cex=1)
lines(x, y, lwd = 3, col = "red")

## Calculating error in Linear Classifier model

error2=0
for (i in 1:200){
  if(data[i,3]==1){
    if((yvalue(est_beta[1]/est_beta[3],est_beta[2]/est_beta[3],data[i,2])<=data[i,1])){
      error2=error2+1
    }
  }
  else{
    if((yvalue(est_beta[1]/est_beta[3],est_beta[2]/est_beta[3],data[i,2])>=data[i,1])){
      error2=error2+1
    }
  }
}

error2/200

## Calculating error in Linear classifier model
## error= (Number of missclassifed datapoints)/(total number of datapoints)

# if y value for (+1) datapoints is more than its value according to classifying line than it will count as error



## Adding a new column to assign its class according to K-NN model
data=cbind(data,matrix(rep(0,200)))



##################################################################################################
## Compute the classifier using k-nearest neighbor method, for k = 15. Plot
## the classifying curve and print the training error


# Calculating the distance of each to every other point 
# Select the least 15 points and assigning 1 if number of (+1) assigned is more than (-1) assigned data points
for(i in 1:200){
  pt_dist=c()
  for(j in 1:200){
    if(i!=j){
      dist=euclidean_dist(data[i,2],data[j,2],data[i,3],data[j,3])
      pt_dist=append(pt_dist,dist)
    }
  }
  first_k=order(pt_dist)
  temp_sum=0
  for(k in 1:15){
    if(data[first_k[k],4]==1){
      temp_sum=temp_sum+1
    }
  }
  if(temp_sum/15>=0.5){
    data[i,5]=1
  }
  else{
    data[i,5]=-1
  }
}


## Calculating the training error in K-NN model
## error= (Number of missclassifed datapoints)/(total number of datapoints)
error3=0
for (i in 1:200){
  if(data[i,4]!=data[i,5]){
    error3=error3+1
  }
}
error3/200
head(data)

## Plot the classifying curve

yy=seq(ymin-0.2,ymax+0.2,length=100)
arr=c()
for (i in 1:100){
  for (j in 1:100){
    ptdist=c()
    temp_arr=c(x[i],yy[j],0)
    for (k in 1:200){
      dist=sqrt((x[i]-data[k,2])**2+(yy[j]-data[k,3])**2)
      ptdist=append(ptdist,dist)
    }
    firstk=order(ptdist)
    tempsum=0
    for(kk in 1:15){
      if(data[firstk[kk],4]==1){
        tempsum=tempsum+1
      }
    }
    if(tempsum/15>=0.5){
      temp_arr[3]=1
    }
    else{
      temp_arr[3]=-1
    }
    arr=append(arr,temp_arr)
  }
}
arr=matrix(arr,10000,3,byrow=TRUE)

plot(aa,bb)
for(i in 1:10000){
  if(arr[i,3]==1){
    points(arr[i,1],arr[i,2],col='yellow')
    ## yellow region refers to class 1
    }
  else{
    points(arr[i,1],arr[i,2],col='pink')
    ## pink region refers to class 1
  }
}
points(data1[1:100,2],data1[1:100,3],col='red',cex=0.7)
points(data2[1:100,2],data2[1:100,3],col='blue',pch=18,cex=1)


##################################################################################################
## Compute the classifier using k-nearest neighbor method, for k = 1. Plot
## the classifying curve and print the training error


## Repeating the same steps of question 3 with k=1
data=cbind(data,matrix(rep(0,200)))
for(i in 1:200){
  ptdist=c()
  for(j in 1:200){
    if(i!=j){
      dist=euclidean_dist(data[i,2],data[j,2],data[i,3],data[j,3])
      ptdist=append(ptdist,dist)
    }
  }
  firstk=order(ptdist)
  tempsum=0
  for(k in 1:1){
    if(data[firstk[k],4]==1){
      tempsum=tempsum+1
    }
  }
  if(tempsum/1>=0.5){
    data[i,6]=1
  }
  else{
    data[i,6]=-1
  }
}


## Calculating training error
## error= (Number of missclassifed datapoints)/(total number of datapoints)


error4=0
for (i in 1:200){
  if(data[i,4]!=data[i,6]){
    error4=error4+1
  }
}
error4/200


## Plotting the classifying curve

arr1=c()
for (i in 1:100){
  for (j in 1:100){
    ptdist1=c()
    temp_arr1=c(x[i],yy[j],0)
    for (k in 1:200){
      dist=sqrt((x[i]-data[k,2])**2+(yy[j]-data[k,3])**2)
      ptdist1=append(ptdist1,dist)
    }
    firstk1=order(ptdist1)
    tempsum1=0
    for(kk in 1:1){
      if(data[firstk1[kk],4]==1){
        tempsum1=tempsum1+1
      }
    }
    if(tempsum1/1>=0.5){
      temp_arr1[3]=1
    }
    else{
      temp_arr1[3]=-1
    }
    arr1=append(arr1,temp_arr1)
  }
}
arr1=matrix(arr1,10000,3,byrow=TRUE)

plot(aa,bb)
for(i in 1:10000){
  if(arr1[i,3]==1){
    ## yellow region refers to class 1
    points(arr1[i,1],arr1[i,2],col='yellow')
    }
  else{
    ## pink region refers to class 1
    points(arr1[i,1],arr1[i,2],col='pink')
  }
}
points(data1[1:100,2],data1[1:100,3],col='red',cex=0.7)
points(data2[1:100,2],data2[1:100,3],col='blue',pch=18,cex=1)

aa

##################################################################################################
## Generate 10000 test vectors as follows: Generate 5000 feature vectors as
## (i.e., 500 vectors for each mi), and label them all as (+1).
## Similarly, generate 5000 more feature vectors (i.e.,
## 500 vectors per mi′ and label them all as (−1)
## Assume that the labels correspond to the true responses
## Compute and print the test error based on the classifiers derived using 
## (i) linear model 
## (ii) 15-NN 
## (iii) 1-NN.


## Repeating same steps with 10000 datapoints

newd1= mvrnorm(n = 10,mu = m1,Sigma = cov)
newvv1=matrix(rep(1,5000))
newvv2=matrix(rep(-1,5000))
newdata1=mvrnorm(n = 500,mu =c(newd1[1,1],newd1[1,2]),Sigma = matrix(c(0.1,0,0,0.1),2,2,byrow=TRUE))

for(i in 2:10){
  newtemp=mvrnorm(n = 500,mu =c(newd1[i,1],newd1[i,2]),Sigma = matrix(c(0.1,0,0,0.1),2,2,byrow=TRUE))
  newdata1=rbind(newdata1,newtemp)
}
newdata1=cbind(newdata1,newvv1)
newdata1=cbind(newvv1,newdata1)


newd2= mvrnorm(n = 10,mu = m2,Sigma = cov)
newdata2=mvrnorm(n = 500,mu =c(newd2[1,1],newd2[1,2]),Sigma = matrix(c(0.1,0,0,0.1),2,2,byrow=TRUE))
for(i in 2:10){
  temp=mvrnorm(n = 500,mu =c(newd2[i,1],newd2[i,2]),Sigma = matrix(c(0.1,0,0,0.1),2,2,byrow=TRUE))
  newdata2=rbind(newdata2,temp)
}
newdata2=cbind(newdata2,newvv2)
newdata2=cbind(newvv1,newdata2)


newdata=rbind(newdata1,newdata2)
dim(newdata)


aa=c(-3,4)
bb=c(-2.5,4)
plot(aa,bb)
points(newdata1[1:5000,2],newdata1[1:5000,3],col='red',cex=0.3)
points(newdata2[1:5000,2],newdata2[1:5000,3],col='blue',pch=18,cex=0.3)

head(newdata[1:10000,1:3])
newtemp_mat=solve(t(newdata[1:10000,1:3])%*%newdata[1:10000,1:3])
newest_beta=(newtemp_mat%*%t(newdata[1:10000,1:3]))%*%newdata[1:10000,4]
dim(newest_beta)

newx=seq(-3,3,length=100)
newy=-(newest_beta[1,1]/newest_beta[3,1])-((newest_beta[2,1]/newest_beta[3,1])*newx)


plot(aa,bb)
points(newdata1[1:5000,2],newdata1[1:5000,3],col='red',cex=0.3)
points(newdata2[1:5000,2],newdata2[1:5000,3],col='blue',pch=18,cex=0.3)
lines(newx, newy, lwd = 3, col = "red")
#set.seed(2525)
#plot(data1[1:100,2],data1[1:100,3],col='red',cex=0.7)
#points(data2[1:100,2],data2[1:100,3],col='blue',pch=18,cex=1)
#axis(1, at = -5:5)
newdata[1,1]
newerror2=0
for (i in 1:10000){
  if(newdata[i,3]==1){
    if((yvalue(newest_beta[1]/newest_beta[3],newest_beta[2]/newest_beta[3],newdata[i,2])<=newdata[i,1])){
      newerror2=newerror2+1
    }
  }
  else{
    if((yvalue(newest_beta[1]/newest_beta[3],newest_beta[2]/newest_beta[3],newdata[i,2])>=newdata[i,1])){
      newerror2=newerror2+1
    }
  }
}

newerror2/10000

newdata=cbind(newdata,matrix(rep(0,10000)))


for(i in 1:10000){
  newpt_dist=c()
  for(j in 1:10000){
    if(i!=j){
      newdist=euclidean_dist(newdata[i,2],newdata[j,2],newdata[i,3],newdata[j,3])
      newpt_dist=append(newpt_dist,newdist)
    }
  }
  newfirst_k=order(newpt_dist)
  newtemp_sum=0
  for(k in 1:15){
    if(newdata[newfirst_k[k],4]==1){
      newtemp_sum=newtemp_sum+1
    }
  }
  if(newtemp_sum/15>=0.5){
    newdata[i,5]=1
  }
  else{
    newdata[i,5]=-1
  }
}
newdata[10,5]


# plot new modifies data



newerror3=0
for (i in 1:10000){
  if(newdata[i,4]!=newdata[i,5]){
    newerror3=newerror3+1
  }
}
newerror3/10000




newdata=cbind(newdata,matrix(rep(0,10000)))
for(i in 1:10000){
  newptdist=c()
  for(j in 1:10000){
    if(i!=j){
      newdist=euclidean_dist(newdata[i,2],newdata[j,2],newdata[i,3],newdata[j,3])
      newptdist=append(newptdist,newdist)
    }
  }
  newfirstk=order(newptdist)
  newtempsum=0
  for(k in 1:1){
    if(newdata[newfirstk[k],4]==1){
      newtempsum=newtempsum+1
    }
  }
  if(newtempsum/1>=0.5){
    newdata[i,6]=1
  }
  else{
    newdata[i,6]=-1
  }
}


# plot new modifies data



newerror4=0
for (i in 1:10000){
  if(newdata[i,4]!=newdata[i,6]){
    newerror4=newerror4+1
  }
}
newerror4/10000
print('end')