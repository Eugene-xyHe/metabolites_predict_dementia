# 1.Survival analysis of individual metabolites and dementia outcomes
fitAD <- coxph( Surv(BL2Target_yrsAD, target_yAD == 1) ~ Metabolite + age + sex + educ + factor(apoe4),data=dat)
summary(fitAD)

fitACD <- coxph( Surv(BL2Target_yrsACD, target_yACD == 1) ~ Metabolite + age + sex + educ + factor(apoe4),data=dat)
summary(fitACD)

fitVaD <- coxph( Surv(BL2Target_yrsVaD, target_yVaD == 1) ~ Metabolite + age + sex + educ + factor(apoe4),data=dat)
summary(fitVaD)





# 2.Survival analysis of MetRS and dementia outcomes
fitAD <- coxph( Surv(BL2Target_yrsAD, target_yAD == 1) ~ MetRS + age + sex + educ + factor(apoe4),data=dat)
summary(fitAD)
fitACD <- coxph( Surv(BL2Target_yrsACD, target_yACD == 1) ~ MetRS + age + sex + educ + factor(apoe4),data=dat)
summary(fitACD)
fitVaD <- coxph( Surv(BL2Target_yrsVaD, target_yVaD == 1) ~ MetRS + age + sex + educ + factor(apoe4),data=dat)
summary(fitVaD)


# 2.1. Survival curves of MetRS and dementia outcomes
dat = dat %>% mutate(MetRS = as.numeric(scale(MetRS)),metRS_3group = factor(ntile(MetRS, 3),levels=c(1,2,3)) )

fit1 <- survfit(Surv(BL2Target_yrs_AD,target_yAD==1) ~ metRS_3group, data = dat) 
p1 <- ggsurvplot(fit1, font.main = c(12, "plain", "darkblue"),fun = "cumhaz",
                       conf.int = T,conf.int.alpha=0.2,
                       palette = c("#2ab8d3","#ee3124"), legend.labs = c('Low',"High"),
                       risk.table = TRUE, risk.table.height =0.2,
                       xlab='Years',ylab=paste0('Cumulative Risk of ',pheno), 
                       xlim = c(0,16), ylim = c(0,myylim[1,pheno]),
                       break.x.by = 4,break.y.by=0.01,
                       surv.plot.height=0.5,censor=F)
					   
fit2 <- survfit(Surv(BL2Target_yrs_ACD,target_yACD==1) ~ metRS_3group, data = dat) 
p2 <- ggsurvplot(fit2, font.main = c(12, "plain", "darkblue"),fun = "cumhaz",
                       conf.int = T,conf.int.alpha=0.2,
                       palette = c("#2ab8d3","#ee3124"), legend.labs = c('Low',"High"),
                       risk.table = TRUE, risk.table.height =0.2,
                       xlab='Years',ylab=paste0('Cumulative Risk of ',pheno), 
                       xlim = c(0,16), ylim = c(0,myylim[1,pheno]),
                       break.x.by = 4,break.y.by=0.01,
                       surv.plot.height=0.5,censor=F)
					   
fit3 <- survfit(Surv(BL2Target_yrs_VaD,target_yVaD==1) ~ metRS_3group, data = dat) 
p3 <- ggsurvplot(fit3, font.main = c(12, "plain", "darkblue"),fun = "cumhaz",
                       conf.int = T,conf.int.alpha=0.2,
                       palette = c("#2ab8d3","#ee3124"), legend.labs = c('Low',"High"),
                       risk.table = TRUE, risk.table.height =0.2,
                       xlab='Years',ylab=paste0('Cumulative Risk of ',pheno), 
                       xlim = c(0,16), ylim = c(0,myylim[1,pheno]),
                       break.x.by = 4,break.y.by=0.01,
                       surv.plot.height=0.5,censor=F)