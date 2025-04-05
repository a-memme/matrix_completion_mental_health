import pandas as pd 
import numpy as np 
import utils as utils 
from sklearn.metrics import matthews_corrcoef, mutual_info_score
from sklearn.preprocessing import MinMaxScaler

class Analysis:

    #Wrangling Functions 
    def wrangle_data(self, mh_df):
        #Make a copy
        df = mh_df.copy()
        df_cols = [utils.snake_case(col) for col in df.columns]
        df.columns = df_cols
        #Define binary cols 
        binary_cols = df.columns[df.apply(lambda col: col.astype(str).str.contains('Yes').any())]
        #Iterate through binary cols and reformat
        for col in binary_cols:
            df[col] = np.where(df[col]=='Yes',1,0)
        
        return df, binary_cols

    #OHE function
    def run_ohe(self, df):
        discrete_vars = df.select_dtypes(include=['object'])
        continuous_vars = df.select_dtypes(exclude=['object'])
        #Apply OHE
        ohe_vars = pd.get_dummies(discrete_vars)
        X_ohe = pd.concat([continuous_vars, ohe_vars], axis=1)

        return X_ohe

    def matthews_df(self, binary_data):
        #Initialize correlation dictionary
        correlation_dict = {}
        #iterate over features as rows
        for row in binary_data.columns:
            if 'index' not in correlation_dict.keys():
                correlation_dict['index'] = [row]
            else:
                correlation_dict['index'].append(row)
            #Features as columns
            for col in binary_data.columns:
                y_true = binary_data[row]
                y_pred = binary_data[col]
                #Calculate matthews coeff
                matthews = matthews_corrcoef(y_true, y_pred)
                if col not in correlation_dict.keys():
                    correlation_dict[col] = [matthews]
                else:
                    correlation_dict[col].append(matthews)
        #Reformat to dataframe
        corr_df = pd.DataFrame(correlation_dict).set_index('index')

        return corr_df
    
    def mutual_information(self, ohe_df):
        #Temporary scaling - MinMax
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(ohe_df), columns=ohe_df.columns, index=ohe_df.index)
        mi_dict = {}
        #iterate over features as rows
        for row in df.columns:
            if 'index' not in mi_dict.keys():
                mi_dict['index'] = [row]
            else:
                mi_dict['index'].append(row)
            #Features as columns
            for col in df.columns:
                y_true = df[row]
                y_pred = df[col]
                #Calculate matthews coeff
                mi = mutual_info_score(y_true, y_pred)
                if col not in mi_dict.keys():
                    mi_dict[col] = [mi]
                else:
                    mi_dict[col].append(mi)
        #Convert to df
        mi_df = pd.DataFrame(mi_dict).set_index('index')

        return mi_df
    
    #Plot Feature Importances
    def plot_fi(self, X, model, model_name, top_n=None):
        feature_columns = X.columns 
        feature_imp = model.feature_importances_

        feature_df = pd.DataFrame({'factors':feature_columns,
                            'importance':feature_imp}).sort_values(['importance'], ascending=False).set_index('factors')
        if top_n is not None:
            feature_df = feature_df[:top_n]

        feature_df.sort_values('importance').plot.barh(y='importance', figsize=(5,5), title=f'{model_name} - Feature Importance')

    