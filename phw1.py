# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 10:23:21 2021

@author: Howoon
"""

'''
<  find_best_model  >
INPUT : config ={        
        'dataframe': pd.DataFrame,
        'encode_cols':columns for encoding  ,
        'scale_cols':columns for encoding,
        'scalers' : scalers list,
        'encoders' : encoders list,
        'models':models list
        'hyperparams': dictionary of hyperparameters }

- Split dataset into trainingset and testset. (0.75:0.25)
- In this process I experienced with DecisionTree classifier, SVM and logistic regressior.
  Also I applied various hyperparameters with grid search and cross validation.

        
OUTPUT : model_combination_list,best_score
          [1] model_combination_list : dataframe of all models combination 
          [2] best_score : best score of testing
'''


def find_best_model(config):
    df = config['dataset']

    # creating features and label 
    X = df.drop('class', axis = 1)
    y = df['class']

    # Model building. Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



    best_accuracy = 0
    best_k=0
    best_model = ''
    #Set model_combination_list dataframe
    comb_lst = pd.DataFrame(columns=['Scaler', 'Encoder', 'Model','K-Fold', 'Score'])
    
    #Experiment with various models
    for model in config['models']:
        #Apply various hyperparameters in each models 
        if model == 'DecisionTree_Gini':
          params_grid=config['hyperparams']['DT_gini_params']
          model=DecisionTreeClassifier()
       
        elif model == 'DecisionTree_Entropy':
          params_grid=config['hyperparams']['DT_entropy_params']
          model=DecisionTreeClassifier()

        elif model == 'SVM':
          params_grid = config['hyperparams']['SVM_params']
          model=SVC()

        elif model == 'LogisticRegression':
          params_grid = config['hyperparams']['LR_params']
          model= LogisticRegression()
        
        # Conduct gridsearch and k-fold cross validation to find best model by test-score
        # Also experiment with various k-values in k-fold cross validation
        for k in  config['hyperparams']['k']:
          kfold = KFold(n_splits=k, shuffle = True, random_state=0)
          grid_search = GridSearchCV(model, params_grid, cv=kfold, n_jobs=-1, refit=True,)
          grid_search.fit(X_train, y_train)
          classifier = grid_search.best_estimator_
          prediction = classifier.predict(X_test)
          accuracy = float(metrics.accuracy_score(y_test, prediction).round(3))
          #Save the model combination and test result
          comb_lst=comb_lst.append({
              'Scaler': config['scaler'],
              'Encoder': config['encoder'],
              'Model':classifier,
              'K-Fold':k,
              'Score':accuracy
              
          }, ignore_index=True)

          if accuracy > best_accuracy:
              best_accuracy = accuracy
              best_model = classifier
              

    return best_accuracy, best_model,comb_lst


'''
<  tuner  >
INPUT : config ={        
        'dataframe': pd.DataFrame,
        'encode_cols':columns for encoding  ,
        'scale_cols':columns for encoding,
        'scalers' : scalers list,
        'encoders' : encoders list,
        'models':models list
        'hyperparams': dictionary of hyperparameters }

- It is like gridsearch. Inspect all combination of each scalers,encoders with models.
        
OUTPUT : model_combination_list,best_score
          [1] model_combination_list : dataframe of all models combination 
          [2] best_score : best score of testing

'''


def tuner(config):
    comb_list = pd.DataFrame(columns=['Scaler', 'Encoder', 'Model','K-Fold', 'Score'])
    best_accuracy = 0;
    #Data preprocessing
    for encoder in config['encoders']:
        temp_df = config['dataframe'].copy()
        # encoding the 'class' columns (belign/maglinant)
        encoding_df = temp_df[config['encode_cols']]
        # if 'NON', belign value is 2 and maglinant value is 4
        if encoder != 'NON':
            for col in config['encode_cols']:
                encoder.fit(temp_df[col])
                encoding_df[col] = encoder.transform(temp_df[col])
        encoding_df=encoding_df.reset_index()

        for scaler in config['scalers']:
            # Scaling the all columns except 'class' feature
            scaled_df = scaler.fit_transform(temp_df[config['scale_cols']])
            scaled_df = pd.DataFrame(scaled_df, columns=config['scale_cols'])
            temp_df = pd.concat([scaled_df, encoding_df], axis=1)
            
            #call find_best_model function to train and test with various models.
            accuracy, model,comb = find_best_model({'dataset': temp_df, 'models': config['models'], 'hyperparams':config['hyperparams'],'scaler':scaler,'encoder':encoder})

            comb_list = comb_list.append(comb, ignore_index=True)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
    # Sort the list in descending order based on the score.
    comb_list=comb_list.sort_values(by=['Score'], axis=0, ascending=False)

    return comb_list, best_accuracy

# find best model with grid search for scaling and encoding
'''
find_best_comb
INPUT : (dataframe , columns to encoding (list) , columns to scaling (list), scalers (list) , encoders (list) ,models (list) )
OUTPUT : (all model combination list (pd.DataFrame) , best_model (dictionary) )

define 'config' dictionary to set list of scalers,encoders,classifiers and hyperparameters.
Call 'tuner()' function to find best model
'''


def find_best_comb(df, encode_col, scale_col, scalers=None, encoders=None, models=None, hyperparams=None):
    # Scalers list (default)
    if scalers == None:
        scalers = [StandardScaler(), MinMaxScaler(),RobustScaler()]
    # Encoders list (default)
    if encoders == None:
        encoders = ['NON', LabelEncoder()]
    # Models list (default)
    if models == None:
        models = [ 'DecisionTree_Gini','DecisionTree_Entropy', 'SVM','LogisticRegression']
    # Hyperparameters list (default)
        if hyperparams==None:
          hyperparams={
              'DT_gini_params':{                 
                  'criterion' : ['gini'],
                  'max_depth' : range(2, 16, 1),
                  'min_samples_leaf' : range(1, 5, 1),
                  'min_samples_split' : range(2, 10, 1),
                  'splitter' : ['best', 'random']
              },
              'DT_entropy_params':{                 
                  'criterion' : ['entropy'],
                  'max_depth' : range(2, 16, 1),
                  'min_samples_leaf' : range(1, 5, 1),
                  'min_samples_split' : range(2, 10, 1),
                  'splitter' : ['best', 'random']
              },
              'SVM_params':{
                  'gamma' : [0.0001, 0.001, 0.01, 0.1],
                  'C' : [0.001, 0.05, 0.5, 0.1, 1, 10, 15, 20]      
              },
              'LR_params':{
                  'C': [ 0.01, 0.1, 1, 10]
              },
              'k':[3,5,7]


          }


    # Setting the parameter 
    config = {
        'dataframe': df,
        'encode_cols': encode_col,
        'scale_cols': scale_col,
        'scalers': scalers,
        'encoders': encoders,
        'models': models,
        'hyperparams' : hyperparams
        }
    
    
    return tuner(config)





def cleaning(df,col_names=None,drop_list=None):
  if col_names==None:
    col_names=['id','clump_thickness','unif_cell_size','unif_cell_shape','marginal_adhesion','single_epith_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
  # concate features' names on dataframe
  df.columns=col_names

  if drop_list==None:
    drop_list=['id']
  # delete irrelevant features
  df.drop(drop_list,axis=1,inplace=True)

  # treating missing values in dataset
  df.replace({'?':np.nan},inplace=True)
  df=df.apply(pd.to_numeric)
  df.dropna(inplace=True)
  #df.fillna(df['bare_nuclei'].mean())
  return df








def main():
    # Load breast_cancer dataset
    df = pd.read_csv("/content/drive/MyDrive/breast_cancer/breast-cancer-wisconsin.csv")

    # setting and cleaning the dataframe 
    df= cleaning(df)

    # finding best model (considering various scalers,encoders, hyperparameters, models...)
    model_combination_list, best_score = find_best_comb(df,encode_col=['class'],scale_col=['clump_thickness','unif_cell_size','unif_cell_shape','marginal_adhesion','single_epith_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses'])

    print(model_combination_list)
    print(best_score)

if __name__ == '__main__':
  main()

rparams=None)
