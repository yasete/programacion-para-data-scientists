def mutual_info(X,y):
    """
    This function finds the correct features to be used.
    X is the feature matrix and y the output corresponding to such matrix.
    X and y must be passed to the function as pandas dataframes
    
    The FIRST CRITERION is based on mutual information. Two terms are considered for each feature:
    the mutual information shared between the feature and the output y (relevancy), and
    the mutual information shared between the feature with the already selected features.
    A stepwise method is used. Only one feature is chosen each time that is included the subset of 
    already selected features.
    The method followed in this function is the "Maximum relevancy, minimum redundancy (MRMR)":
    for each feature x_i it is calculated I(x_i,y)-alpha * sum(I(x_i,x_j)) where x_j is every feature
    already selected. The feature selected in each step is that with a maximum in the previous quantity.
    
    The function returns [S,mutual_info], where S is the set of features to be selected and mutual_info
    the quantity corresponding to every feature.
    
    """

    ###### LIBRARIES #########################
    import sklearn
    import numpy as np
    import pandas as pd
    from sklearn import datasets
    from sklearn import metrics
    import warnings
    warnings.filterwarnings('ignore')
    ###########################################
    
    n=len(np.transpose(X))
    print('The number of potential features is',n)
    S=[] # Set of relevant features
    S_max=[]
         

    # First I calculate the first most relevant feature
    max_mutual_info=0
    mutual_table=[]
    for cont in range(0,n):
            mutual_info=sklearn.metrics.mutual_info_score(y.iloc[:,0],X.iloc[:,cont]) 
            mutual_table.append(mutual_info)
            if mutual_info>max_mutual_info:
                max_mutual_info=mutual_info
                relevant_index=cont
    S.append(relevant_index)
    features_out=[x for x in range(0, n) if x not in S] 
    print('\n\nThis is the table of mutual information:',mutual_table)
    print('\n\nThe most correlated (with y) feature is: ', relevant_index)
     
    
   
    #### Calculation of the rest of features ############
      
    for t in range(0,n):
        alpha=1/(t+1)
        max_mutual_info=0
        for cont_out in features_out:
            mutual_info=sklearn.metrics.mutual_info_score(y.iloc[:,0],X.iloc[:,cont_out])
            for cont_in in range(0,n):
                if cont_out!=cont_in:
                    if cont_in in features_out:
                        mutual_info-=alpha*sklearn.metrics.mutual_info_score(X.iloc[:,cont_out],X.iloc[:,cont_in])
                        if mutual_info>max_mutual_info:
                            max_mutual_info=mutual_info
                            relevant_index=cont_out
        
        S.append(relevant_index)
        S_max.append(max_mutual_info)
        features_out=[x for x in range(0, n) if x not in S]  

        
    #Now we eliminate the duplicates in S
    not_repeated=[]
    for cont in S:
        processed=[x for x in not_repeated]
        if cont not in processed:
            not_repeated.append(cont)
    
   
    S=not_repeated
    print('\n\n Next the subset of relevant features and their corresponding weights')
    print(S)   
    print(S_max)
    return[S,S_max]


def split(data,features,target,percentage_train,percentage_cv):
    """
    This function splits the data in the corresponding x_train, y_train, x_cv, y_cv, x_test, y_test, x_total, y_total by using
    the parameter frac to assign which percentage needs to go for the training data set
    Parameters: data, features, target, percentage_train and percentage_cv (percentage on the remaining data once stracted training set.
    The function gives back [data_train,data_test,x_train,y_train,x_test,y_test,x_total,y_total]
    """

    ###### LIBRARIES #########################
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    import warnings
    warnings.filterwarnings('ignore')
    ###########################################
    
    #### DATA SPLIT #############
    data_train=data.sample(frac=percentage_train,replace=False)
    data_remaining=data[~data.index.isin(data_train.index)]
    data_cv=data_remaining.sample(frac=percentage_cv,replace=False)
    data_test=data_remaining[~data_remaining.index.isin(data_cv.index)]
    
    ######## MATRIX BUILDING #################
    
    x_train=data_train[features]
    y_train=data_train[target]
    x_cv=data_cv[features]
    y_cv=data_cv[target]
    x_test=data_test[features]
    y_test=data_test[target]
    x_total=data[features]
    y_total=data[target]
    
    return [data_train,data_cv,data_test,x_train,y_train,x_cv,y_cv,x_test,y_test,x_total,y_total]

def model_calibrated(x,y,model,cv,verbose):
  '''
  x: x_total matrix
  y: y_total vector
  cv: CV strategy
  verbose: if set to 1 it prints brier loss. Otherwise it is not printed
  output: [model tuned,brier score]
  
  This function tunes a model to provide better probabilities.
  The model passed as parameter does not need to be fitted first.
  Brier score is used for that.
  Both 'isotonic' and 'sigmoid' calibration methods are used. It is choosen the
  one giving a lower Brier score.
  The calibrator is used with CV (any CV strategy can be passsed as parameter CV)
  It prints the final brier score once the model has been tuned. 
  
  
  '''
  from sklearn.calibration import CalibratedClassifierCV as calibrator
  from sklearn.metrics import brier_score_loss
  
  calibration='sigmoid'
  cali = calibrator(model,method=calibration,cv=cv)
  cali.fit(x, y) 
  yhat=cali.predict_proba(x)
  brier_sigmoid=brier_score_loss(y,yhat[:,1])
  
  calibration='isotonic'
  cali = calibrator(model,method=calibration,cv=cv)
  cali.fit(x, y) 
  yhat=cali.predict_proba(x)
  
  brier_isotonic=brier_score_loss(y,yhat[:,1])
  
  if brier_sigmoid < brier_isotonic:
    cali = calibrator(model,method='sigmoid',cv=cv)
    cali.fit(x, y) 
    yhat=cali.predict_proba(x)
    brier_final=brier_score_loss(y,yhat[:,1])
  else:
    cali = calibrator(model,method='isotonic',cv=cv)
    cali.fit(x, y) 
    yhat=cali.predict_proba(x)
    brier_final=brier_score_loss(y,yhat[:,1])
  if verbose==1:
      print('The brier loss once the model is optimized is:',brier_final,'\n')
  return [cali,brier_final]