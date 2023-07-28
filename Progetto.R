### --------------------------------------- Logistica -----------------------------------------------------

## Da fare:
# - ipotesi per usare anova finale
# - interpretazione anova -> https://stats.stackexchange.com/questions/83040/comparison-of-two-logistic-regression-models-significant-result-with-anova-bu
# - interpretare binned-plot

library(caTools) # Librerie per la regressione logistica (o multinomiale)
library(caret) # cross validation library
library(pROC) # libreria per la ROC
library(PRROC) # libreria per la ROC
library(AICcmodavg) # Libreria per il confronto con l'AIC
library(corrplot) # Libreria per la matrice di correlazione
library(ResourceSelection) # Libreria per l'Hoslem Test
library(MASS) # per fare lo stepAIC
library(arm) # per il binned plot
library(rms)
library(car)

library(ggplot2)

# Importazione dataset con header per nome colonne
obes= read.csv("Obesity.csv", header= TRUE)
head(obes)

## ----------------------------------------- Data cleaning --------------------------------------------

# Togliamo virgola da età: 
obes[, c(2)] = trunc(obes[, c(2)])

# Pulizia covariate categoriche 
obes$male <- ifelse(obes$Gender == "Male", 1, 0)
obes= subset(obes,select= -c(Gender))

obes$pos_smoke <- ifelse(obes$SMOKE == "yes", 1, 0)
obes= subset(obes,select= -c(SMOKE))

obes$fam_history <- ifelse(obes$family_history_with_overweight == "yes", 1, 0)
obes= subset(obes,select= -c(family_history_with_overweight))

obes$CALC = factor(obes$CALC, levels = c("Always", "Frequently", "Sometimes", "no"), ordered = TRUE)
obes$CALC <- as.numeric(obes$CALC)

obes$moto <- ifelse(obes$MTRANS == "Motorbike", 1, ifelse(obes$MTRANS == "Automobile", 1, ifelse(obes$MTRANS == "Public_Transportation", 1, 0)))
# Tolto perchè ha correlazione negativa con moto
# obes$walk <- ifelse(obes$MTRANS == "Walking", 1, ifelse(obes$MTRANS == "Bike", 1, 0))
obes= subset(obes,select= -c(MTRANS))

obes$FAVC <- ifelse(obes$FAVC == "no", 0, 1)

obes$CAEC = factor(obes$CAEC, levels = c("Always", "Frequently", "Sometimes", "no"), ordered = TRUE)
obes$CAEC <- as.numeric(obes$CAEC)

obes$SCC <- ifelse(obes$SCC == "no", 0, 1)

# Normalizzazione dei dati per migliorare il binned plot ed il training del nostro modello - possibile confronto con modello non normalizzato
# obes$Age = obes$Age/mean(obes$Age)
# obes$Height = obes$Height/mean(obes$Height) 
# obes$Weight = obes$Weight/mean(obes$Weight)
# obes$FAVC = obes$FAVC/mean(obes$FAVC)
# obes$NCP = obes$NCP/mean(obes$NCP)
# obes$CAEC = obes$CAEC/mean(obes$CAEC)
# obes$CH2O = obes$CH2O/mean(obes$CH2O)
# obes$FAF = obes$FAF/mean(obes$FAF)
# obes$TUE = obes$TUE/mean(obes$TUE)
# obes$CALC = obes$CALC/mean(obes$CALC)

# per una logistica semplice metto a 1 qualsiasi disfunzione di peso 0 altrimenti
obes$overweight_obesity <- ifelse(obes$NObeyesdad == "Normal_Weight", 0, 1)
obes= subset(obes,select= -c(NObeyesdad))

## ----------------------------------------- Data Preparation --------------------------------------------

# Verified assumption on data - link utile: 
# https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-logistic-regression/

## Splitting the dataset to do cross-validation
set.seed(1)
sample <- sample(c(TRUE, FALSE), nrow(obes), replace=TRUE, prob=c(0.7,0.3))
train  <- obes[sample, ]
test   <- obes[!sample, ]

## 3 - Multicollinearity - spiegazione: codingprof.com/3-ways-to-test-for-multicollinearity-in-r-eACC_SENS_SPECamples/
# il corrplot davanti indica la libreria perchè altrimenti usa il corrplot della libreria arm ed è brutto
corrplot::corrplot(cor(obes), method = "number")

## ----------------------------------------- Models Training --------------------------------------------

prob = 0.83 # probabilità di sbarramento per scindere tra 0 e 1

# Training model - 1
logistic_model <- glm(overweight_obesity ~ .,
                      data = train,
                      family = "binomial")
pred_train = ifelse(predict(logistic_model, train, type = "response") > prob, 1, 0)
pred_test= ifelse(predict(logistic_model, test, type = "response") > prob, 1, 0)

# Training model - 2 -> solo cofattori con almeno 1 stellina
logistic_model_2 <- glm(overweight_obesity ~ Weight + FAVC + CAEC +FAF + male + pos_smoke + moto,
                      data = train,
                      family = "binomial")
pred_train_2 = ifelse(predict(logistic_model_2, train, type = "response") > prob, 1, 0)
pred_test_2 = ifelse(predict(logistic_model_2, test, type = "response") > prob, 1, 0)


# Training model - 3 -> solo cofattori con almeno 2 stelline
logistic_model_3 <- glm(overweight_obesity ~ Weight + CAEC + male + pos_smoke + moto,
                        data = train,
                        family = "binomial")
pred_train_3  = ifelse(predict(logistic_model_3, train, type = "response") > prob, 1, 0)
pred_test_3 = ifelse(predict(logistic_model_3, test, type = "response") > prob, 1, 0)

# Training model - 4 -> solo cofattori con 3 stelline
logistic_model_4 <- glm(overweight_obesity ~ Weight + CAEC + male + moto,
                        data = train,
                        family = "binomial")
pred_train_4  = ifelse(predict(logistic_model_4, train, type = "response") > prob, 1, 0)
pred_test_4 = ifelse(predict(logistic_model_4, test, type = "response") > prob, 1, 0)

# Training model - 5 -> modello ottimizzato con la funzione stepAIC che cerca il modello che ottimizza l'AIC
# link utii: http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/154-stepwise-regression-essentials-in-r/
logistic_model_5 <- stepAIC(logistic_model, direction = "both")
pred_train_5  = ifelse(predict(logistic_model_5, train, type = "response") > prob, 1, 0)
pred_test_5 = ifelse(predict(logistic_model_5, test, type = "response") > prob, 1, 0)
summary(logistic_model_5)

## ----------------------------------------- Comparison of models - GOF --------------------------------------------
# link utili:
# - https://bookdown.org/jefftemplewebb/IS-6489/logistic-regression.html#:~:text=In%20logistic%20regression%2C%20as%20with,regression%20are%20generally%20not%20useful.
# - https://atm.amegroups.org/article/view/10171/11129#residual-plotting

# AIC comparison - sito utile per interpretazione: 
# - https://www.scribbr.com/statistics/akaike-information-criterion/
# - https://www.statology.org/what-is-a-good-aic-value/

models <- list(logistic_model, logistic_model_2, logistic_model_3, logistic_model_4, logistic_model_5 )
model.names <- c('features_0_stars', 'features_1_stars', 'features_2_stars','features_3_stars', 'StepAIC_model')

aictab(cand.set = models, modnames = model.names)

# Confusion Matrix
test_tab = table(predicted = pred_test, actual = test$overweight_obesity)
test_con_mat = confusionMatrix(test_tab, positive = "1")

test_tab_2 = table(predicted = pred_test_2, actual = test$overweight_obesity)
test_con_mat_2 = confusionMatrix(test_tab_2, positive = "1")

test_tab_3 = table(predicted = pred_test_3, actual = test$overweight_obesity)
test_con_mat_3 = confusionMatrix(test_tab_3, positive = "1")

test_tab_4 = table(predicted = pred_test_4, actual = test$overweight_obesity)
test_con_mat_4 = confusionMatrix(test_tab_4, positive = "1")

test_tab_5 = table(predicted = pred_test_5, actual = test$overweight_obesity)
test_con_mat_5 = confusionMatrix(test_tab_5, positive = "1")

# creazione matrice di accuretezza, specificità, sensibilità
ACC_SENS_SPEC <- matrix(1:15, ncol = 3, dimnames = list(c('features_0_stars', 'features_1_stars', 'features_2_stars','features_3_stars', 'StepAIC_model'), c("Accuracy","Sensitivity","Specificity")))
# logistic_model
ACC_SENS_SPEC[1, 1] = test_con_mat$overall["Accuracy"]
ACC_SENS_SPEC[1, 2] = test_con_mat$byClass["Sensitivity"]
ACC_SENS_SPEC[1, 3] = test_con_mat$byClass["Specificity"]

# logistic_model_2
ACC_SENS_SPEC[2, 1] = test_con_mat_2$overall["Accuracy"]
ACC_SENS_SPEC[2, 2] = test_con_mat_2$byClass["Sensitivity"]
ACC_SENS_SPEC[2, 3] = test_con_mat_2$byClass["Specificity"]

# logistic_model_3
ACC_SENS_SPEC[3, 1] = test_con_mat_3$overall["Accuracy"]
ACC_SENS_SPEC[3, 2] = test_con_mat_3$byClass["Sensitivity"]
ACC_SENS_SPEC[3, 3] = test_con_mat_3$byClass["Specificity"]

# logistic_model_4
ACC_SENS_SPEC[4, 1] = test_con_mat_4$overall["Accuracy"]
ACC_SENS_SPEC[4, 2] = test_con_mat_4$byClass["Sensitivity"]
ACC_SENS_SPEC[4, 3] = test_con_mat_4$byClass["Specificity"]

# logistic_model_5
ACC_SENS_SPEC[5, 1] = test_con_mat_5$overall["Accuracy"]
ACC_SENS_SPEC[5, 2] = test_con_mat_5$byClass["Sensitivity"]
ACC_SENS_SPEC[5, 3] = test_con_mat_5$byClass["Specificity"]

# Matrice ordinata per accuratezza crescente e poi per sensibilità crescente
ACC_SENS_SPEC[order(ACC_SENS_SPEC[, 1], ACC_SENS_SPEC[, 2]), ]

## ROC
PRROC_obj <- roc.curve(scores.class0 = logistic_model$fitted.values, weights.class0=obes$overweight_obesity, curve=TRUE)
PRROC_obj_2 <- roc.curve(scores.class0 = logistic_model_2$fitted.values, weights.class0=obes$overweight_obesity, curve=TRUE)
PRROC_obj_3 <- roc.curve(scores.class0 = logistic_model_3$fitted.values, weights.class0=obes$overweight_obesity, curve=TRUE)
PRROC_obj_4 <- roc.curve(scores.class0 = logistic_model_4$fitted.values, weights.class0=obes$overweight_obesity, curve=TRUE)
PRROC_obj_5 <- roc.curve(scores.class0 = logistic_model_5$fitted.values, weights.class0=obes$overweight_obesity, curve=TRUE)

graphics.off()

par(mfrow = c(2,2))
plot(PRROC_obj)
plot(PRROC_obj_2)
plot(PRROC_obj_3)
plot(PRROC_obj_4)
plot(PRROC_obj_5)

## Analisi ANOVA per capire se la differenza informativa tra modelli con covariate diverse è trascurabile o meno
anova( logistic_model_2, logistic_model, test = "Chisq" ) # confronto tra modello con ogni covariata ed quello con features da alameno 1 stella
anova( logistic_model_2, logistic_model_5, test = "Chisq" ) # confronto tra modello con ogni covariata ed quello con features da alameno 1 stella

# Interpretazione: 
# 1 - l'aggiunta di tutte le covariate (logistic_models) che porta all'aumento della complessità non è giustificato da una migliore accuratezza;
# 2 - l'aggiunta delle covariate ed il conseguente aumento della commplessità è giustificato da un miglioramento delle performance del modello in accuratezza
# info prese da: https://bookdown.org/ndphillips/YaRrr/comparing-regression-models-with-anova.html

## coeff
# https://quantifyinghealth.com/interpret-logistic-regression-coefficients/#:~:text=The%20logistic%20regression%20coefficient%20%CE%B2,the%20outcome%20by%20e%CE%B2.

coeff <- exp(logistic_model_5$coefficients)
coeff

barplot(coeff, las = 2, cex.names = 0.7,
        col= ifelse(coeff < 1,"red","green"))
abline(a = 1, b = 0, lty = 2)
