import pandas as pd
import numpy as np 
from scipy import stats
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,confusion_matrix, classification_report, mean_absolute_error, mean_squared_error


class ML:

    #Continuous vars
    def _test_continuous(self, continuous_cols, sparse_df, continuous_df, test_df, calc_total=False):
        y_pred_list = []
        y_true_list = []
        rmse=None
        #Continuous vars 
        print("Continuous Metrics:\n")
        for col in continuous_cols:
            ix = np.where(sparse_df[col].isna())
            y_pred = continuous_df[col].iloc[ix]
            y_true = test_df[col].iloc[ix]
            #Column metrics
            print(f"{col} Metrics:")
            print(f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred))}")
            print(f"MAE = {mean_absolute_error(y_true, y_pred)}\n")
            #Append 
            y_pred_list += y_pred.tolist() 
            y_true_list += y_true.tolist()
        if calc_total is True:
            print("\nTotal Continuous:")
            rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))
            print(f"RMSE = {rmse}")
            print(f"MAE = {mean_absolute_error(y_true_list, y_pred_list)}\n")
        
        return rmse

    #Discrete 
    def _test_discrete(self, discrete_cols, sparse_df, discrete_df, test_df):
        #Discrete vars
        #Store each col pairs in lists for final performance calculations
        y_pred_list = []
        y_true_list = []
        #Iterate over columns and calculate scores for imputed vals
        for col in discrete_cols:
            ix = np.where(sparse_df[col].isna())
            y_pred = discrete_df[col].iloc[ix]
            y_true = test_df[col].iloc[ix]
            #Append 
            y_pred_list += y_pred.tolist() 
            y_true_list += y_true.tolist()
        #Score 
        f1 = f1_score(y_true_list, y_pred_list)

        #Print Metrics
        print("Discrete Metrics:\n")
        print(f"F1 Score:{f1}")
        print(f"Accuracy Score = {accuracy_score(y_true_list, y_pred_list)}")
        print(f"Recall = {recall_score(y_true_list, y_pred_list)}")
        print(f"Precision = {precision_score(y_true_list, y_pred_list)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_true_list, y_pred_list)}\n")

        return f1

    def _eval_data(self, threshold, X_pred, discrete_cols, continuous_cols, scaler):
        #Round
        pred_df = X_pred.copy()
        for col in discrete_cols:
            pred_df[col] = np.where(pred_df[col]>threshold, 1, 0)
        #Reverse transform
        eval = pd.DataFrame(scaler.inverse_transform(pred_df), columns=pred_df.columns, index=pred_df.index)
        #Seperate into discrete and continuous dfs
        eval_d = eval[discrete_cols]
        eval_c = eval[continuous_cols]

        return eval_d, eval_c
    

    #KNN Imputation validation 
    def knn_imputer_tuning(self, kmin, kmax, step, X_train, X_test, thresh, discrete_cols, continuous_cols, scaler, 
                           calc_total=False):
        best_score = 0
        best_k = None

        best_rmse = 1000
        c_k = None
        #Iterate over k range
        for k in range(kmin, kmax, step):
            #Print
            print(f"Performance metrics for k = {k}:\n")
            #KNNImputer instance 
            k_imputer = KNNImputer(n_neighbors=k)
            #Predictions in a df and round vals
            knn_continuous = pd.DataFrame(k_imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            #Round discrete column values to binary 0,1
            knn_df = knn_continuous.copy()
            for col in discrete_cols:
                knn_df[col] = np.where(knn_df[col]>thresh, 1, 0)
            #Reverse transform 
            eval = pd.DataFrame(scaler.inverse_transform(knn_df), columns=knn_df.columns, index=knn_df.index)
            #Discrete and Continuous 
            knn_d = eval[discrete_cols]
            knn_c = eval[continuous_cols]
            
            #Discrete vars
            if len(discrete_cols) > 0:
                f1 = self._test_discrete(discrete_cols=discrete_cols, sparse_df=X_train, discrete_df=knn_d, test_df=X_test)
                #Update score
                if f1 > best_score:
                    best_score = f1
                    best_k = k

            #Continuous vars 
            if len(continuous_cols) > 0:
                rmse = self._test_continuous(continuous_cols=continuous_cols, sparse_df=X_train, continuous_df=knn_c, test_df=X_test, calc_total=calc_total)
                if calc_total is True:
                    if rmse < best_rmse:
                        best_rmse = rmse 
                        c_k = k
                    print(f"Best continuous k = {c_k} with rmse = {best_rmse}")

        return best_score, best_k

    #MICE Imputation validation 
    def mice_imputer_tuning(self, estimators, inits, order, sps, X_train, X_test, thresh, discrete_cols, continuous_cols, scaler, 
                        calc_total=False):
        best_score = 0
        best_params = {}

        best_rmse = 1000
        c_params = {}

        for e in estimators:
            for i in inits:
                for o in order:
                    for sp in sps:
                        #Impute
                        mice = IterativeImputer(max_iter=100, min_value=0, max_value=1, 
                                                estimator = e, initial_strategy=i, imputation_order=o, sample_posterior=sp)
                        mice_continuous = pd.DataFrame(mice.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                        #Reverse transform and segment 
                        mice_d, mice_c = self._eval_data(threshold=thresh, X_pred=mice_continuous, 
                                                discrete_cols=discrete_cols, continuous_cols=continuous_cols, 
                                                scaler=scaler)
                        #Discrete vars
                        if len(discrete_cols) > 0:
                            f1 = self._test_discrete(discrete_cols=discrete_cols, sparse_df=X_train, discrete_df=mice_d, test_df=X_test)
                            #Update score
                            if f1 > best_score:
                                best_score = f1
                                best_params['estimator'] = e
                                best_params['initial_strategy'] = i
                                best_params['imputation_order'] = o
                                best_params['sample_posterior'] = sp
                            print(f"Best discrete params = {best_params} with f1 = {best_score}")
                        #Continuous vars 
                        if len(continuous_cols) > 0:
                            rmse = self._test_continuous(continuous_cols=continuous_cols, sparse_df=X_train, continuous_df=mice_c, test_df=X_test, calc_total=calc_total)
                            if calc_total is True:
                                if rmse < best_rmse:
                                    best_rmse = rmse 
                                    c_params['estimator'] = e
                                    c_params['initial_strategy'] = i
                                    c_params['imputation_order'] = o
                                    c_params['sample_posterior'] = sp
                                print(f"Best continuous params = {c_params} with rmse = {best_rmse}")

        return best_score, best_params
    
    #Custom Singular Value Thresholding algorithm
    def singular_value_thresholding(self, M:np.array, init, tau, delta, max_iter, tol):
        #Create matrix copy
        X = M.copy()
        #Create mask
        mask = ~np.isnan(M)
        #Imputation initialization 
        if init == 'mode':
            impute = stats.mode(M, axis=0, nan_policy='omit')[0]
        elif init == 'mean':
            impute = np.nanmean(M, axis=0)
        else:
            raise ValueError("imputation initialization must be either 'mean' or 'mode'")
        #Fill NaNs
        X[~mask] = np.take(impute, np.where(~mask)[1]).flatten()
        # for i in range(X.shape[1]):
        #     X[np.isnan(M[:, i]), i] = impute[i]
        #Iterate over max_iter
        for iteration in range(max_iter):
            #SVD
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            #Thresholding operation - threshold singular values
            S_thresh = np.maximum(S - tau, 0)
            #Reconstruct matrix using S_thresh
            X_new = U @ np.diag(S_thresh) @ Vt
            #Only update missing entries using mask (leave regular values in tact)
            X_new[mask] = M[mask]
            #Add nonnegativity constraint
            X_new = np.maximum(X_new, 0)
            # Update X with weighted step size delta 
            # delta < 1 means more conservative update (blending majority of previous X with X_new)
            X = delta * X_new + (1 - delta) * X
            #Manually check for convergence via euclidean distance
            if np.linalg.norm(X - X_new, 'fro') / np.linalg.norm(X, 'fro') < tol:
                print(f"Converged in {iteration + 1} iterations")
                break
            #Update X
            X = X_new
        
        return X
    
    #SVT Tuning 
    def svt_tuning(self, X_train, inits, taus, deltas, X_test, thresh, discrete_cols, continuous_cols, scaler, calc_total=False):
        best_score = 0
        best_params = {}

        best_rmse = 1000
        c_params = {}

        for i in inits:
            for t in taus:
                for d in deltas:
                    #Print
                    print(f"\nPerformance metrics for init={i}; tau={t}; delta={d}\n")
                    #Run svt
                    X = self.singular_value_thresholding(M=np.array(X_train), init=i, tau=t, delta=d, max_iter=100000, tol=1e-5)
                    #Convert numpy array to dataframe
                    svt_continuous = pd.DataFrame(X, columns=X_train.columns, index=X_train.index)
                    #Round
                    svt_df = svt_continuous.copy()
                    #Apply rounding for binary vals
                    for col in discrete_cols:
                        svt_df[col] = np.where(svt_df[col]>thresh, 1, 0)
                    #Reverse transform 
                    eval = pd.DataFrame(scaler.inverse_transform(svt_df), columns=svt_df.columns, index=svt_df.index)

                    #Discrete and Continuous 
                    svt_d = eval[discrete_cols]
                    svt_c = eval[continuous_cols]
                    
                    #Discrete vars
                    if len(discrete_cols) > 0:
                        f1 = self._test_discrete(discrete_cols=discrete_cols, sparse_df=X_train, discrete_df=svt_d, test_df=X_test)
                        #Update score
                        if f1 > best_score:
                            best_score = f1
                            best_params['init'] = i
                            best_params['tau'] = t
                            best_params['delta'] = d
                        print(f"Best discrete params = {best_params} \nwith f1 = {best_score}")

                    #Continuous vars 
                    if len(continuous_cols) > 0:
                        rmse = self._test_continuous(continuous_cols=continuous_cols, sparse_df=X_train, continuous_df=svt_c, test_df=X_test, calc_total=calc_total)
                        if calc_total is True:
                            if rmse < best_rmse:
                                best_rmse = rmse 
                                c_params['init'] = i
                                c_params['tau'] = t
                                c_params['delta'] = d
                            print(f"Best continuous params = {c_params} \nwith rmse = {best_rmse}")

        return best_score, best_params