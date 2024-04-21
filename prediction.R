
Outcome <- 'Surv(BL2Target_yrs,target_y==1)~'
c1 <- c("age",'sex',"educ","apoe4")
cog <- c("pm_time", "rt_time")

FML1<- as.formula(paste(Outcome,paste(c('MetRS'),collapse=" + ")))
FML8<- as.formula(paste(Outcome,paste(c('MetRS',c1),collapse=" + ")))
FML9<- as.formula(paste(Outcome,paste(c('MetRS',c1,cog),collapse=" + ")))

FML <- list(FML1,FML8,FML9)

covar <- as.data.frame(fread('./final_Covariates_full_population.tsv.gz')) 
categorical <- c("sex","smk","apoe4","diab_hst","cvd_hst","med_bp","med_tc")
for (i in categorical){  covar[,i] <- as.factor(covar[,i]) }

test <- train <- list()
fold=1
for (fold in 1:10){
  test[[fold]] <- as.data.frame(fread( paste0('./MetRS/TestFold',fold-1,'/testing.csv')  ))
  test[[fold]] <- merge(test[[fold]],covar,by='eid')
  test[[fold]]$BL2Target_yrs_original=test[[fold]]$BL2Target_yrs
  train[[fold]] <- as.data.frame(fread(   paste0('./MetRS/TestFold',fold-1,'/training.csv')   )) 
  train[[fold]] <- merge(train[[fold]],covar,by='eid')
  cox <-  list()
  for (i in 1:3){
	cox[[i]]<- coxph(FML[[i]],data=train[[fold]])
	yrs=15   #yrs代表想要预测的随访时间	
	   test[[fold]]$BL2Target_yrs <- yrs
	   test[[fold]][,paste0('pred_Model',i,'_years',yrs)]<- predict(cox[[i]],test[[fold]],type="lp")   }}
test_all <-plyr::rbind.fill(test[1:10]) 
test_all[,paste0('target_y_binary',yrs)] <- ifelse(test_all$BL2Target_yrs_original>yrs,0,test_all$target_y)
model <- fit_test <- list()
for (i in 1:3){
	fit_test[[i]] <- roc(as.formula(paste0('target_y_binary',yrs,' ~ pred_Model',i,'_years',yrs)),
					 test_all,ci=T,auc = T)
	AUC=as.numeric(fit_test[[i]]$ci)[2]
	AUC_LCI=as.numeric(fit_test[[i]]$ci)[1]
	AUC_UCI=as.numeric(fit_test[[i]]$ci)[3]
	model[[i]] <- data.frame('Model'= paste0('Model ',i),
						   'Times'= yrs,
						   'AUC'=AUC,
						   'AUC_95CI'=sprintf("%.4f-%.4f",min(AUC_LCI,AUC_UCI),max(AUC_LCI,AUC_UCI)    )  )}