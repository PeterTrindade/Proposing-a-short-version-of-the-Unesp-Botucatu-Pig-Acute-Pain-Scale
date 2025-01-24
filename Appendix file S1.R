#Random Forest (RF)

#Importing packages

pacotes <- c(
  'tidyverse',  
  'rpart',      
  'rpart.plot', 
  'gtools',     
  'Rmisc',      
  'scales',     
  'viridis',    
  'caret',       
  'AMR',
  'randomForest',
  'fastDummies',
  'rattle',
  'xgboost',
  'ggeffects',
  'spaMM',
  'dfoptim',
  'patchwork',
  'gridExtra',
  'grid',
  'lme4',
  'pROC',
  'bestglm',
  'glmmTMB',
  'biostatUZH',
  'irr',
  "dplyr",
  'ggplot2',
  'RColorBrewer'
  )

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}

#Cleaninig the global environment

rm(list=ls())

#Sample size computation
power.roc.test(auc=.90, sig.level=0.05, power=0.95)

#Reliability

#Importing database (Appendix file S2)
d100=read.csv2("Appendix file S2.csv",h=T,d=",",na.strings=".")
str(d100)

#Dummy
d100=dummy_columns(d100,select_columns=c(  
  'Posture',
  'Interaction',
  'Activity',
  'Lift.Pelvic.Limb',
  'Scratching.Rubbing',
  'Walk.Away.Run',
  'Sit.With.Difficulty',
  'Continuously.Balances',
  'Bite.Grill',
  'Head.Down',
  'Difficulty.Overcoming'
),
remove_selected_columns=F,
remove_first_dummy=F)

#Inter-observer reliability of the UPAPS total sum (ICC)
icc(matrix(c(
  d100$Total.Score.Sum[d100$Observers=="Observer.1"],
  d100$Total.Score.Sum[d100$Observers=="Observer.2"]),nrow=90,ncol=2),model="twoway",unit="average",
  type="consistency",conf.level=0.95)

icc(matrix(c(
  d100$Total.Score.Sum[d100$Observers=="Observer.1"],
  d100$Total.Score.Sum[d100$Observers=="Observer.3"]),nrow=90,ncol=2),model="twoway",unit="average",
  type="consistency",conf.level=0.95)

icc(matrix(c(
  d100$Total.Score.Sum[d100$Observers=="Observer.2"],
  d100$Total.Score.Sum[d100$Observers=="Observer.3"]),nrow=90,ncol=2),model="twoway",unit="average",
  type="consistency",conf.level=0.95)

icc(matrix(c(
  d100$Total.Score.Sum[d100$Observers=="Observer.4"],
  d100$Total.Score.Sum[d100$Observers=="Observer.5"]),nrow=28,ncol=2),model="twoway",unit="average",
  type="consistency",conf.level=0.95)

#Train–test split – training base
#Importing database (Appendix file S2)

d100=read.csv2("Appendix file S2.csv",h=T,d=",",na.strings=".")
d70=filter(d100,Split=='Train')

#Creating dummy variables for training base

d70=dummy_columns(d70,select_columns=c(  
  'Posture',
  'Interaction',
  'Activity',
  'Lift.Pelvic.Limb',
  'Scratching.Rubbing',
  'Walk.Away.Run',
  'Sit.With.Difficulty',
  'Continuously.Balances',
  'Bite.Grill',
  'Head.Down',
  'Difficulty.Overcoming'
),
remove_selected_columns=F,
remove_first_dummy=F)

train <- d70[,c('Condition',
                 'Posture_Posture.1',
                 'Posture_Posture.2',
                 'Posture_Posture.3',
                 'Interaction_Interaction.1',
                 'Interaction_Interaction.2',
                 'Interaction_Interaction.3',
                 'Activity_Activity.1',
                 'Activity_Activity.2',
                 'Activity_Activity.3',
                 'Lift.Pelvic.Limb_Lift.Pelvic.Limb.1',
                 'Scratching.Rubbing_Scratching.Rubbing.1',
                 'Walk.Away.Run_Walk.Away.Run.1',
                 'Sit.With.Difficulty_Sit.With.Difficulty.1',
                 'Continuously.Balances_Continuously.Balances.1',
                 'Bite.Grill_Bite.Grill.1',
                 'Head.Down_Head.Down.1',
                 'Difficulty.Overcoming_Difficulty.Overcoming.1')]

train$Condition=factor(train$Condition, labels = c('N','Y'))

#Train–test split – testing base
#Importing database (Appendix file S2)

d100=read.csv2("Appendix file S2.csv",h=T,d=",",na.strings=".")  
d30=filter(d100, Split=='Test')

#Creating dummy variables for testing base

d30=dummy_columns(d30,select_columns=c(  
  'Posture',
  'Interaction',
  'Activity',
  'Lift.Pelvic.Limb',
  'Scratching.Rubbing',
  'Walk.Away.Run',
  'Sit.With.Difficulty',
  'Continuously.Balances',
  'Bite.Grill',
  'Head.Down',
  'Difficulty.Overcoming'
),
remove_selected_columns=F,
remove_first_dummy=F)

test <- d30[,c('Condition',
                'Posture_Posture.1',
                'Posture_Posture.2',
                'Posture_Posture.3',
                'Interaction_Interaction.1',
                'Interaction_Interaction.2',
                'Interaction_Interaction.3',
                'Activity_Activity.1',
                'Activity_Activity.2',
                'Activity_Activity.3',
                'Lift.Pelvic.Limb_Lift.Pelvic.Limb.1',
                'Scratching.Rubbing_Scratching.Rubbing.1',
                'Walk.Away.Run_Walk.Away.Run.1',
                'Sit.With.Difficulty_Sit.With.Difficulty.1',
                'Continuously.Balances_Continuously.Balances.1',
                'Bite.Grill_Bite.Grill.1',
                'Head.Down_Head.Down.1',
                'Difficulty.Overcoming_Difficulty.Overcoming.1')]

test$Condition=factor(test$Condition, labels = c('N','Y'))

#Training Random Forest

#Customizing grid search for optimize (tuning) the random forest

customRF <- list(type = "Classification",
                 library = "randomForest",
                 loop = NULL)

customRF$parameters <- data.frame(parameter = c("mtry","ntree","number","repeats"),
                                  class = rep("numeric", 4),
                                  label = c("mtry", "ntree","number","repeats"))

customRF$grid <- function(x, y, len = NULL, search = "grid") {}

customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs) {
  randomForest(x, y,
               mtry = param$mtry,
               ntree=param$ntree,
               number=param$number,
               repeats=param$repeats)
}

#Creating a predict label for the Random Forest

customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
predict(modelFit, newdata)

#Creating a predict probability for the Random Forest

customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

#Optimizing (tuning) the random forest

set.seed(2360873)
control <- trainControl(method="repeatedcv", 
                        number=2, 
                        repeats=2,
                        allowParallel=T)
set.seed(2360873)
tunegrid <- expand.grid(.mtry=c(2:17),.ntree=c(501,1001,2001),.number=c(2:10),.repeats=c(2:10))

set.seed(2360873)
custom <- train(Condition~
                  +Posture_Posture.1
                +Posture_Posture.2
                +Posture_Posture.3
                +Interaction_Interaction.1
                +Interaction_Interaction.2
                +Interaction_Interaction.3
                +Activity_Activity.1
                +Activity_Activity.2
                +Activity_Activity.3
                +Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                +Scratching.Rubbing_Scratching.Rubbing.1
                +Walk.Away.Run_Walk.Away.Run.1
                +Sit.With.Difficulty_Sit.With.Difficulty.1
                +Continuously.Balances_Continuously.Balances.1
                +Bite.Grill_Bite.Grill.1
                +Head.Down_Head.Down.1
                +Difficulty.Overcoming_Difficulty.Overcoming.1,
                data=train, 
                method=customRF, 
                metric="Accuracy", 
                tuneGrid=tunegrid, 
                trControl=control)

summary(custom)
print(custom)
varImp(custom)

#Extracting variable importance

VI=varImp(custom)

#Importing database for plotting importances (Appendix file S3)

df1=read.csv2("Appendix file S3.csv",h=T,d=",", na.strings = ".")
df1 = mutate(df1, X = case_when(
  X =='Posture_Posture.1' ~ "Posture 1",
  X =='Posture_Posture.2' ~ "Posture 2",
  X =='Posture_Posture.3' ~ "Posture 3",
  X =='Interaction_Interaction.1' ~ "Interaction 1",
  X =='Interaction_Interaction.2' ~ "Interaction 2",
  X =='Interaction_Interaction.3' ~ "Interaction 3",
  X =='Activity_Activity.1' ~ "Activity 1",
  X =='Activity_Activity.2' ~ "Activity 2",
  X =='Activity_Activity.3' ~ "Activity 3",
  X =='Lift.Pelvic.Limb_Lift.Pelvic.Limb.1' ~ "Elevates pelvic limb",
  X =='Scratching.Rubbing_Scratching.Rubbing.1' ~ "Scratches/rubs painful area",
  X =='Walk.Away.Run_Walk.Away.Run.1' ~ "Moves away/runs away",
  X =='Sit.With.Difficulty_Sit.With.Difficulty.1' ~ "Sit with difficulty",
  X =='Continuously.Balances_Continuously.Balances.1' ~ "Wags tail",
  X =='Bite.Grill_Bite.Grill.1' ~ "Bites bar or objects",
  X =='Head.Down_Head.Down.1' ~ "Head down",
  X =='Difficulty.Overcoming_Difficulty.Overcoming.1' ~ "Difficulty overcoming obstacles",
))
df1$N=round(df1$N,2)

tiff("VariableImportanceRF2.tiff",width=10,height=5,units='in',res=600,
     compression = c("none", "rle", "lzw", "jpeg", "zip", "lzw+p", "zip+p"),
     family="sans")
ggplot(df1,aes(x=N,y=reorder(X, -N),label=N))+ 
  geom_col(color="#a65c85ff",fill="#a65c85ff")+
  scale_x_continuous(n.breaks=10)+ 
  ylab("UPAPS pain-altered behaviors") +
  xlab("Importance based on random forest algorithm (%)")+
  scale_color_viridis_d(begin=.2,end=.8)+
  theme_classic()+theme(axis.text=element_text(size=12),
                        axis.title=element_text(size=12),
                        legend.text=element_text(size=12),
                        legend.title=element_text(size=12),
                        panel.grid.major=element_blank(),
                        plot.margin = unit(c(1,1,1,1),"cm"))+
  geom_text(aes(label = N),vjust =.45,hjust=-.1)
dev.off()

#ROC curve

train$Probability=predict(custom,train,type="prob")[,2]
myroc_train=pROC::roc(Condition~Probability,train);myroc_train$auc
set.seed(2360873)
pROC::ci.auc(Condition~Probability,train)

set.seed(2360873)
citt=ci.coords(roc(Condition~Probability,train,plot=F,
                   algorithm=2,smooth=F,boot.n=1001,boot.stratified=T,ci.auc=T,auc=T),x="best",
               input=c("threshold", "specificity", "sensitivity"),
               ret=c("threshold", "specificity", "sensitivity"),
               best.method="youden", 
               best.policy = "random", 
               conf.level=0.95, boot.n=1001,
               boot.stratified=TRUE) 

write.csv2(citt,'CI_train.csv')

#Testing Random Forest

#ROC curve using algorithm probability

test$Probability=predict(custom,test,type="prob")[,2]  
myroc_test=pROC::roc(Condition~Probability,test);myroc_test$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test)  

set.seed(2360873)
cite=ci.coords(roc(Condition~Probability,test,plot=F,
                   algorithm=2,smooth=F,boot.n=1001,boot.stratified=T,ci.auc=T,auc=T),x="best",
               input=c("threshold", "specificity", "sensitivity"),
               ret=c("threshold", "specificity", "sensitivity"),
               best.method="youden",
               best.policy = "random",
               conf.level=0.95, boot.n=1001,
               boot.stratified=TRUE)

write.csv2(cite,'CI_test.csv')

#ROC curve using UPAPS total sum

myroc_upaps=pROC::roc(Condition~Total.Score.Sum,d30);myroc_upaps$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Total.Score.Sum,d30)

set.seed(2360873)
cis=ci.coords(roc(Condition~Total.Score.Sum,d30,plot=F,
                  algorithm=2,smooth=F,boot.n=1001,boot.stratified=T,ci.auc=T,auc=T),x="best",
              input=c("threshold", "specificity", "sensitivity"),
              ret=c("threshold", "specificity", "sensitivity"),
              best.method="youden",
              best.policy = "random",
              conf.level=0.95, boot.n=1001,
              boot.stratified=TRUE)

write.csv2(cis,'CI_UPAPS.csv')

roc.test(myroc_train,myroc_test)
roc.test(myroc_test,myroc_upaps)


#Refining Ramdom Forest

#Random forest using from 1st to 16th best behaviors

control_sel <- caret::trainControl(
  method='repeatedcv',
  number=4, 
  repeats=9,
  search='grid', 
  summaryFunction = twoClassSummary, 
  classProbs = TRUE 
)

grid_sel <- base::expand.grid(.mtry=2) 

set.seed(2360873)
custom1_16 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2
                    +Posture_Posture.1
                    +Walk.Away.Run_Walk.Away.Run.1
                    +Interaction_Interaction.1
                    +Bite.Grill_Bite.Grill.1
                    +Activity_Activity.3
                    +Posture_Posture.3
                    +Activity_Activity.2
                    +Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    +Scratching.Rubbing_Scratching.Rubbing.1
                    +Interaction_Interaction.3,
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

print(custom1_16)

test$Probability=predict(custom1_16,test,type="prob")[,2]
myroc_test1_16=pROC::roc(Condition~Probability,test);myroc_test1_16$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_16,myroc_test)

#Random forest using from 1st to 15th best behaviors

set.seed(2360873)
custom1_15 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2
                    +Posture_Posture.1
                    +Walk.Away.Run_Walk.Away.Run.1
                    +Interaction_Interaction.1
                    +Bite.Grill_Bite.Grill.1
                    +Activity_Activity.3
                    +Posture_Posture.3
                    +Activity_Activity.2
                    +Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    +Scratching.Rubbing_Scratching.Rubbing.1,
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_15,test,type="prob")[,2]
myroc_test1_15=pROC::roc(Condition~Probability,test);myroc_test1_15$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_15,myroc_test)

#Random forest using from 1st to 14th best behaviors

set.seed(2360873)
custom1_14 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2
                    +Posture_Posture.1
                    +Walk.Away.Run_Walk.Away.Run.1
                    +Interaction_Interaction.1
                    +Bite.Grill_Bite.Grill.1
                    +Activity_Activity.3
                    +Posture_Posture.3
                    +Activity_Activity.2
                    +Lift.Pelvic.Limb_Lift.Pelvic.Limb.1,
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_14,test,type="prob")[,2]
myroc_test1_14=pROC::roc(Condition~Probability,test);myroc_test1_14$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_14,myroc_test)

#Random forest using from 1st to 13th best behaviors

set.seed(2360873)
custom1_13 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2
                    +Posture_Posture.1
                    +Walk.Away.Run_Walk.Away.Run.1
                    +Interaction_Interaction.1
                    +Bite.Grill_Bite.Grill.1
                    +Activity_Activity.3
                    +Posture_Posture.3
                    +Activity_Activity.2,
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1,  
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_13,test,type="prob")[,2]
myroc_test1_13=pROC::roc(Condition~Probability,test);myroc_test1_13$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_13,myroc_test)

#Random forest using from 1st to 12th best behaviors

set.seed(2360873)
custom1_12 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2
                    +Posture_Posture.1
                    +Walk.Away.Run_Walk.Away.Run.1
                    +Interaction_Interaction.1
                    +Bite.Grill_Bite.Grill.1
                    +Activity_Activity.3
                    +Posture_Posture.3,
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1,  
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_12,test,type="prob")[,2]
myroc_test1_12=pROC::roc(Condition~Probability,test);myroc_test1_12$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_12,myroc_test)

#Random forest using from 1st to 11th best behaviors

set.seed(2360873)
custom1_11 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2
                    +Posture_Posture.1
                    +Walk.Away.Run_Walk.Away.Run.1
                    +Interaction_Interaction.1
                    +Bite.Grill_Bite.Grill.1
                    +Activity_Activity.3,
                    #+Posture_Posture.3
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1,  
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_11,test,type="prob")[,2]
myroc_test1_11=pROC::roc(Condition~Probability,test);myroc_test1_11$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_11,myroc_test)

#Random forest using from 1st to 10th best behaviors

set.seed(2360873)
custom1_10 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2
                    +Posture_Posture.1
                    +Walk.Away.Run_Walk.Away.Run.1
                    +Interaction_Interaction.1
                    +Bite.Grill_Bite.Grill.1,
                    #+Activity_Activity.3
                    #+Posture_Posture.3
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_10,test,type="prob")[,2]
myroc_test1_10=pROC::roc(Condition~Probability,test);myroc_test1_10$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_10,myroc_test)

#Random forest using from 1st to 9th best behaviors

set.seed(2360873)
custom1_09 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2
                    +Posture_Posture.1
                    +Walk.Away.Run_Walk.Away.Run.1
                    +Interaction_Interaction.1,
                    #+Bite.Grill_Bite.Grill.1
                    #+Activity_Activity.3
                    #+Posture_Posture.3
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_09,test,type="prob")[,2]
myroc_test1_09=pROC::roc(Condition~Probability,test);myroc_test1_09$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_09,myroc_test)

#Random forest using from 1st to 8th best behaviors

set.seed(2360873)
custom1_08 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2
                    +Posture_Posture.1
                    +Walk.Away.Run_Walk.Away.Run.1,
                    #+Interaction_Interaction.1,
                    #+Bite.Grill_Bite.Grill.1
                    #+Activity_Activity.3
                    #+Posture_Posture.3
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_08,test,type="prob")[,2]
myroc_test1_08=pROC::roc(Condition~Probability,test);myroc_test1_08$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_08,myroc_test)

#Random forest using from 1st to 7th best behaviors

set.seed(2360873)
custom1_07 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2
                    +Posture_Posture.1,
                    #+Walk.Away.Run_Walk.Away.Run.1,
                    #+Interaction_Interaction.1,
                    #+Bite.Grill_Bite.Grill.1
                    #+Activity_Activity.3
                    #+Posture_Posture.3
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_07,test,type="prob")[,2]
myroc_test1_07=pROC::roc(Condition~Probability,test);myroc_test1_07$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_07,myroc_test)

#Random forest using from 1st to 6th best behaviors

set.seed(2360873)
custom1_06 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1
                    +Posture_Posture.2,
                    #+Posture_Posture.1,
                    #+Walk.Away.Run_Walk.Away.Run.1,
                    #+Interaction_Interaction.1,
                    #+Bite.Grill_Bite.Grill.1
                    #+Activity_Activity.3
                    #+Posture_Posture.3
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_06,test,type="prob")[,2]
myroc_test1_06=pROC::roc(Condition~Probability,test);myroc_test1_06$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_06,myroc_test)

#Random forest using from 1st to 5th best behaviors

set.seed(2360873)
custom1_05 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2
                    +Activity_Activity.1,
                    #+Posture_Posture.2,
                    #+Posture_Posture.1,
                    #+Walk.Away.Run_Walk.Away.Run.1,
                    #+Interaction_Interaction.1,
                    #+Bite.Grill_Bite.Grill.1
                    #+Activity_Activity.3
                    #+Posture_Posture.3
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_05,test,type="prob")[,2]
myroc_test1_05=pROC::roc(Condition~Probability,test);myroc_test1_05$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_05,myroc_test)

#Random forest using from 1st to 4th best behaviors

set.seed(2360873)
custom1_04 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1
                    +Interaction_Interaction.2,
                    #+Activity_Activity.1,
                    #+Posture_Posture.2,
                    #+Posture_Posture.1,
                    #+Walk.Away.Run_Walk.Away.Run.1,
                    #+Interaction_Interaction.1,
                    #+Bite.Grill_Bite.Grill.1
                    #+Activity_Activity.3
                    #+Posture_Posture.3
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1,  
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_04,test,type="prob")[,2]
myroc_test1_04=pROC::roc(Condition~Probability,test);myroc_test1_04$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_04,myroc_test)

#Random forest using from 1st to 3th best behaviors

set.seed(2360873)
custom1_03 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1
                    +Continuously.Balances_Continuously.Balances.1,
                    #+Interaction_Interaction.2,
                    #+Activity_Activity.1,
                    #+Posture_Posture.2,
                    #+Posture_Posture.1,
                    #+Walk.Away.Run_Walk.Away.Run.1,
                    #+Interaction_Interaction.1,
                    #+Bite.Grill_Bite.Grill.1
                    #+Activity_Activity.3
                    #+Posture_Posture.3
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_03,test,type="prob")[,2]
myroc_test1_03=pROC::roc(Condition~Probability,test);myroc_test1_03$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_03,myroc_test)

#Random forest using from 1st to 2nd best behaviors

set.seed(2360873)
custom1_02 <- train(Condition~
                      +Sit.With.Difficulty_Sit.With.Difficulty.1
                    +Head.Down_Head.Down.1,
                    #+Continuously.Balances_Continuously.Balances.1,
                    #+Interaction_Interaction.2,
                    #+Activity_Activity.1,
                    #+Posture_Posture.2,
                    #+Posture_Posture.1,
                    #+Walk.Away.Run_Walk.Away.Run.1,
                    #+Interaction_Interaction.1,
                    #+Bite.Grill_Bite.Grill.1
                    #+Activity_Activity.3
                    #+Posture_Posture.3
                    #+Activity_Activity.2
                    #+Lift.Pelvic.Limb_Lift.Pelvic.Limb.1
                    #+Scratching.Rubbing_Scratching.Rubbing.1
                    #+Interaction_Interaction.3
                    #+Difficulty.Overcoming_Difficulty.Overcoming.1, 
                    data=train, 
                    method='rf', 
                    metric="ROC", 
                    tuneGrid=grid_sel,
                    ntree=501,
                    trControl=control_sel)

test$Probability=predict(custom1_02,test,type="prob")[,2]
myroc_test1_02=pROC::roc(Condition~Probability,test);myroc_test1_02$auc 
set.seed(2360873)
pROC::ci.auc(Condition~Probability,test) 

roc.test(myroc_test1_02,myroc_test)

#Ploting random forest refining AUCs

#Importing database for plotting random forest refining AUCs (Appendix file S4)

dauc=read.csv2("Appendix file S4.csv",h=T,d=",",na.strings=".")

unique(dauc$Ranking)

dauc$Ranking=factor(dauc$Ranking,levels = c('17.16',
                                            '17.15',
                                            '17.14',
                                            '17.13',
                                            '17.12',
                                            '17.11',
                                            '17.10',
                                            '17.09',
                                            '17.08',
                                            '17.07',
                                            '17.06',
                                            '17.05',
                                            '17.04',
                                            '17.03',
                                            '17.02'
),
labels = c(
  '1st to 16th',
  '1st to 15th',
  '1st to 14th',
  '1st to 13th',
  '1st to 12th',
  '1st to 11th',
  '1st to 10th',
  '1st to 9th',
  '1st to 8th',
  '1st to 7th',
  '1st to 6th',
  '1st to 5th',
  '1st to 4th',
  '1st to 3rd',
  '1st to 2nd'
))
str(dauc)

tiff("AUC_refing.tiff",width=6,height=5,units='in',res=300,
     compression = c("none", "rle", "lzw", "jpeg", "zip", "lzw+p", "zip+p"),
     family="sans")
ggplot(dauc,aes(y=Ranking,x=AUC, fill=Ranking,color=Ranking))+ 
  geom_vline(aes(xintercept=91.38),size=.4,linetype="dashed",color='gray20')+ 
  geom_pointrange(aes(xmax=High,xmin=Low),
                  position=position_dodge(width=.5),
                  size=1,shape=21,show.legend = F)+
  scale_x_continuous(n.breaks=15,limits = c(70,100))+
  xlab("AUC") +
  ylab("Best-ranked UPAPS pain-altered behaviors")+
  theme_bw()+theme(axis.text=element_text(size=10),
                   axis.title=element_text(size=12),
                   legend.text=element_text(size=12),
                   legend.title=element_text(size=12),
                   panel.grid.major.y=element_blank(),
                   panel.grid.major.x=element_blank(),
                   legend.position = 'bottom')+
  
  scale_fill_viridis_d(begin=.0,end=.85)+
  scale_color_viridis_d(begin=.0,end=.85)+
  
  annotate("text",x=94.84,y=13,label="*",angle=0,color='black',size=6)+
  annotate("text",x=92.65,y=14,label="*",angle=0,color='black',size=6)+
  annotate("text",x=87.5,y=15,label="*",angle=0,color='black',size=6)+
  geom_rect(aes(ymin = 11.5, ymax = 12.5, xmin = 70, xmax = 100),color='gray20',fill="coral4",alpha=0,size=1)
dev.off()


#Short pain scale

#Calculation a total sum for the Short UPAPS

d30$UPAPS_short=rowSums(d30[c('Sit.With.Difficulty_Sit.With.Difficulty.1',
                               'Head.Down_Head.Down.1',
                               'Continuously.Balances_Continuously.Balances.1',
                               'Interaction_Interaction.2',
                               'Activity_Activity.1')])

# ROC curve
myroc_upaps_short=pROC::roc(Condition~UPAPS_short,d30);myroc_upaps_short$auc 
set.seed(2360873)
pROC::ci.auc(Condition~UPAPS_short,d30)

set.seed(2360873)
cis=ci.coords(roc(Condition~UPAPS_short,d30,plot=F,
                  algorithm=2,smooth=F,boot.n=1001,boot.stratified=T,ci.auc=T,auc=T),x="best",
              input=c("threshold", "specificity", "sensitivity"),
              ret=c("threshold", "specificity", "sensitivity"),
              best.method="youden",
              best.policy = "random",
              conf.level=0.95, boot.n=1001,
              boot.stratified=TRUE)

write.csv2(cis,'CI_UPAPS_short.csv')

roc.test(myroc_upaps,myroc_upaps_short)
