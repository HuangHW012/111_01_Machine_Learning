import numpy as np
import pandas as pd
import evaluation
from scipy.constants import speed_of_light
# 繪圖
import matplotlib.pyplot as plt


# 訓練測試拆分
from sklearn.model_selection import train_test_split
# 標準化
from sklearn.preprocessing import StandardScaler
# 降維
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 機器學習方法
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
# 管線設計
from sklearn.pipeline import make_pipeline
# 學習曲線評估
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import classification_report, roc_curve, auc, RocCurveDisplay
# 找超參數
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
# 計算時間
import time


if __name__ == '__main__':
    add_new_feature_flag = True
    train_flag = False
    constraints_flag = train_flag
    all_model_flag = False
    
    print("Load the training/test data using pandas")
    train = pd.read_csv("./tau_data/training.csv")
    check_agreement = pd.read_csv('./tau_data/check_agreement.csv', index_col='id')
    check_correlation = pd.read_csv('./tau_data/check_correlation.csv', index_col='id')
    
    if add_new_feature_flag == True:
        # 新增特徵
        train['p0_pz'] = np.sqrt(train['p0_p']**2 - train['p0_pt']**2)
        train['p1_pz'] = np.sqrt(train['p1_p']**2 - train['p1_pt']**2)
        train['p2_pz'] = np.sqrt(train['p2_p']**2 - train['p2_pt']**2)
        train['pz'] = train['p0_pz'] + train['p1_pz'] + train['p2_pz']
        train['p'] = np.sqrt(train['pt']**2 + train['pz']**2)
        train['measure_speed'] = (train['FlightDistance'] / train['LifeTime']) # [mm / ns]
        #train['distance_ratio_error'] = (train['FlightDistanceError'] / train['FlightDistance'])
        #test['pseudo_speed'] = (test['FlightDistance'] / test['LifeTime']) * (1e6)
        #train['speed'] = np.sqrt(np.square(train['pseudo_speed']) / (1 + np.square(train['pseudo_speed']/speed_of_light)))
        #train['approx_mass'] =  train['p'] / (train['measure_speed'] / (speed_of_light * 1e-6))
        
        
        check_agreement['p0_pz'] = (check_agreement['p0_p']**2 - check_agreement['p0_pt']**2)**0.5
        check_agreement['p1_pz'] = (check_agreement['p1_p']**2 - check_agreement['p1_pt']**2)**0.5
        check_agreement['p2_pz'] = (check_agreement['p2_p']**2 - check_agreement['p2_pt']**2)**0.5
        check_agreement['pz'] = check_agreement['p0_pz'] + check_agreement['p1_pz'] + check_agreement['p2_pz']
        check_agreement['p'] = (check_agreement['pt']**2 + check_agreement['pz']**2)**0.5
        #check_agreement['speed'] = check_agreement['FlightDistance'] / check_agreement['LifeTime']
        check_agreement['measure_speed'] = (check_agreement['FlightDistance'] / check_agreement['LifeTime']) # [mm / ns]
        #check_agreement['pseudo_speed'] = (check_agreement['FlightDistance'] / check_agreement['LifeTime']) * (1e6)
        #check_agreement['speed'] = np.sqrt(np.square(check_agreement['pseudo_speed']) / (1 + np.square(check_agreement['pseudo_speed'] / speed_of_light)))
        #check_agreement['approx_mass'] =  check_agreement['p'] / (check_agreement['measure_speed'] / (speed_of_light * 1e-6))
        
        check_correlation['p0_pz'] = (check_correlation['p0_p']**2 - check_correlation['p0_pt']**2)**0.5
        check_correlation['p1_pz'] = (check_correlation['p1_p']**2 - check_correlation['p1_pt']**2)**0.5
        check_correlation['p2_pz'] = (check_correlation['p2_p']**2 - check_correlation['p2_pt']**2)**0.5
        check_correlation['pz'] = check_correlation['p0_pz'] + check_correlation['p1_pz'] + check_correlation['p2_pz']
        check_correlation['p'] = (check_correlation['pt']**2 + check_correlation['pz']**2)**0.5
        #check_correlation['speed'] = check_correlation['FlightDistance'] / check_correlation['LifeTime']
        check_correlation['measure_speed'] = (check_correlation['FlightDistance'] / check_correlation['LifeTime']) # [mm / ns]
        #check_correlation['pseudo_speed'] = (check_correlation['FlightDistance'] / check_correlation['LifeTime']) * (1e6)
        #check_correlation['speed'] = np.sqrt(np.square(check_correlation['pseudo_speed']) / (1 + np.square(check_correlation['pseudo_speed']/speed_of_light)))
        #check_correlation['approx_mass'] =  check_correlation['p'] / (check_correlation['measure_speed'] / (speed_of_light * 1e-6))
    
        # 去除 SPDhits
        features = list(train.columns[1:-5-6]) + list(train.columns[-6:])
    else:
        features = list(train.columns[1:-4])
        
        
    # 分割訓練、測試    
    X = train[features]
    y = train['signal']   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state = 42)   
    
    # 標準化數據
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    
    # 特徵降維 (PCA)
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_std)
    # 繪圖使用
    pca_095 = PCA(n_components = 0.95)
    pca_095.fit_transform(X_train_std)
    
    xi = np.arange(1, X_train_std.shape[1] + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    # 產生圖片
    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, X_train_std.shape[1] + 1, step = 5))
    plt.yticks(np.arange(0, 1.2, step = 0.2), labels = np.arange(0, 120, step = 20))
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.98, '95%', color = 'red', fontsize=16)
    plt.axvline(x=pca_095.n_components_, color='g', linestyle='-')
    plt.text(26, -0.1, f'{pca_095.n_components_}', color = 'g', fontsize=16)
    plt.grid()
    plt.show()

    if train_flag == True:
        # svm, lr, dt, rf, xgb, ada, knearest
        model_type = 'xgb'

        if model_type == 'svm':
            #model = SVC(random_state = 42)
            model = BaggingClassifier(SVC(random_state = 42, kernel = 'linear'))
            model_name = 'Support Vector Machine'
        elif model_type == 'lr':    
            model = LogisticRegression(random_state = 42, max_iter=1000)
            model_name = 'Logistic Regression'
        elif model_type == 'dt':    
            model = DecisionTreeClassifier(random_state = 42)
            model_name = 'Decision Tree'
        elif model_type == 'rf':    
            model = RandomForestClassifier(random_state = 42)
            model_name = 'Random Forest'
        elif model_type == 'xgb':    
            model = XGBClassifier(random_state = 42, max_depth = 2)
            model_name = 'XGBoost'
        elif model_type == 'ada':    
            model = AdaBoostClassifier(random_state = 42)
            model_name = 'AdaBoost'
        elif model_type == 'knearest':    
            model = KNeighborsClassifier()
            model_name = 'K-neighbors'
           
        # 管線設計
        pipe = make_pipeline(StandardScaler(),\
                             #PCA(n_components = 0.95), \
                             model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print('Test Accuracy: %.3f' % pipe.score(X_test, y_test))
        
        
        
        # 驗證曲線
        if model_type == 'svm':
            param_name = 'svc__C'
            param_range = [1.0]
        elif model_type == 'lr':    
            param_name = 'logisticregression__C'
            param_range = [0.01, 0.1, 1, 10, 100]
        elif model_type == 'dt':    
            param_name = 'decisiontreeclassifier__max_depth'
            param_range = np.arange(1, 15, 1)
        elif model_type == 'rf':    
            param_name = 'randomforestclassifier__max_depth'
            param_range = np.arange(1, 15, 1)
            #param_name = 'randomforestclassifier__n_estimators'
            #param_range = np.arange(50, 100, 10)
        elif model_type == 'xgb':    
            #param_name = 'xgbclassifier__max_depth'
            #param_range = np.arange(1, 15, 1)
            param_name = 'xgbclassifier__n_estimators'
            param_range = np.arange(50, 200, 10)
        elif model_type == 'ada':    
            param_name = 'adaboostclassifier__n_estimators'
            param_range = np.arange(10, 100, 10)
            #param_name = 'adaboostclassifier__learning_rate'
            #param_range = np.arange(0.1, 2, 0.5)
        elif model_type == 'knearest':    
            param_name = 'kneighborsclassifier__n_neighbors'
            param_range = np.arange(1, 15, 1)
            
            
        train_scores, test_scores = validation_curve(estimator=pipe, X=X_train, y=y_train, \
                                                     param_name=param_name, param_range=param_range, cv=5, n_jobs=-1)
                                                                
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        
        
        plt.title(f"Validation Curve ({model_name})")
        plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
        plt.fill_between(param_range,test_mean + test_std,test_mean - test_std, alpha=0.15, color='green')
        plt.grid()
        plt.legend(loc='lower right')
        plt.xlabel('Parameter')
        plt.ylabel('Accuracy')
        #plt.xscale("log")
        plt.ylim([0.6, 1.03])
        plt.show()
        
        
        
        
        
        
        # 學習曲線
        if model_type == 'svm':
            #model = SVC(random_state = 42)
            best_model = BaggingClassifier(SVC(random_state = 42, kernel = 'linear'))
        elif model_type == 'lr':    
            best_model = LogisticRegression(random_state = 42, max_iter=1000)
        elif model_type == 'dt':    
            best_model = DecisionTreeClassifier(random_state = 42, max_depth = 6) 
        elif model_type == 'rf':    
            best_model = RandomForestClassifier(random_state = 42, max_depth = 6)
        elif model_type == 'xgb':    
            best_model = XGBClassifier(random_state = 42, max_depth = 2)
        elif model_type == 'ada':    
            best_model = AdaBoostClassifier(random_state = 42, n_estimators = 50)
        elif model_type == 'knearest':    
            best_model = KNeighborsClassifier(n_neighbors = 6)

            
        # 管線設計
        best_pipe = make_pipeline(StandardScaler(),\
                             #PCA(n_components = 0.95), \
                             best_model)
        best_pipe.fit(X_train, y_train)
        y_pred = best_pipe.predict(X_test)
        print('(best) Test Accuracy: %.3f' % best_pipe.score(X_test, y_test))    
        train_sizes, train_scores, test_scores = learning_curve(estimator=best_pipe, X=X_train, y=y_train, \
                                                                train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=-1)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # 學習曲線
        plt.title(f"Learning Curve ({model_name})")
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.yticks(np.arange(0.8, 0.95, step = 0.025))
        plt.grid()
        plt.show()
        
        
        print('[訓練]')
        train_y_pred = best_pipe.predict(X_train)
        print(classification_report(y_train, train_y_pred, digits = 4))
        fpr, tpr, _ = roc_curve(y_train, train_y_pred, pos_label=1)
        print(f'AUC: {auc(fpr, tpr)}')
        
        print('[測試]')
        test_y_pred = best_pipe.predict(X_test)
        print(classification_report(y_test, test_y_pred, digits = 4))
        fpr, tpr, _ = roc_curve(y_test, test_y_pred, pos_label=1)
        print(f'AUC: {auc(fpr, tpr)}')
        
        
        # 測試
        test = pd.read_csv("./tau_data/test.csv")
        if add_new_feature_flag == True:
            test['p0_pz'] = np.sqrt(test['p0_p']**2 - test['p0_pt']**2)
            test['p1_pz'] = np.sqrt(test['p1_p']**2 - test['p1_pt']**2)
            test['p2_pz'] = np.sqrt(test['p2_p']**2 - test['p2_pt']**2)
            test['pz'] = test['p0_pz'] + test['p1_pz'] + test['p2_pz']
            test['p'] = np.sqrt(test['pt']**2 + test['pz']**2)
            test['measure_speed'] = (test['FlightDistance'] / test['LifeTime']) # [mm / ns]
            
        test_predict_probs = best_pipe.predict_proba(test[features])[:,1]
        submission = pd.DataFrame({"id": test["id"], "prediction": test_predict_probs})
        submission.to_csv("xgboost_submission.csv", index=False)
        
        
        
        
        
        # 管線設計 + 超參數找尋 (GridSearchCV)
        
        pipe_hyper = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
        #distributions = dict(decisiontreeclassifier__max_depth = [1, 2, 3, 4, 5])
        clf = RandomizedSearchCV(pipe_hyper, param_distributions=[{'randomforestclassifier__max_depth': np.arange(1, 10, 1)},\
                                                  {'randomforestclassifier__n_estimators': np.arange(80, 100, 10)},\
                                                  ], random_state=0)
        #clf = GridSearchCV(pipe_hyper,  param_grid=[{'randomforestclassifier__max_depth': np.arange(1, 10, 1)},\
        #                                          {'randomforestclassifier__n_estimators': np.arange(80, 100, 10)},\
        #                                          ], scoring='accuracy')
        search = clf.fit(X_train, y_train)
        
        
        scores = cross_val_score(search, X_train, y_train, scoring='accuracy', cv=5)
        print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
        
    if constraints_flag == True:     

        # Check agreement test
        agreement_probs = best_pipe.predict_proba(check_agreement[features])[:,1]
        
        ks = evaluation.compute_ks(
            agreement_probs[check_agreement['signal'].values == 0],
            agreement_probs[check_agreement['signal'].values == 1],
            check_agreement[check_agreement['signal'] == 0]['weight'].values,
            check_agreement[check_agreement['signal'] == 1]['weight'].values)
        print('KS metric', ks, ks < 0.09)
        
        # Check correlation test
        correlation_probs = best_pipe.predict_proba(check_correlation[features])[:,1]
        
        cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
        print('CvM metric', cvm, cvm < 0.002)
        
        # Compute weighted AUC on the training data with min_ANNmuon > 0.4
        train_eval = train[train['min_ANNmuon'] > 0.4]
        train_probs = best_pipe.predict_proba(train_eval[features])[:,1]
        AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
        print('AUC', AUC)
        
        
        
    if all_model_flag == True: 
        # 管線設計
        # Logistic Regression
        best_pipe_LR = make_pipeline(StandardScaler(),\
                             LogisticRegression(random_state = 42, max_iter=1000))
        start = time.time()
        best_pipe_LR.fit(X_train, y_train)
        end = time.time()
        print(f'LR: {end - start} s')
        best_pipe_LR_PCA = make_pipeline(StandardScaler(),\
                             PCA(n_components = 0.95), \
                             LogisticRegression(random_state = 42, max_iter=1000))
        start = time.time()
        best_pipe_LR_PCA.fit(X_train, y_train)
        end = time.time()
        print(f'LR(pca): {end - start} s')
        # DT
        best_pipe_DT = make_pipeline(StandardScaler(),\
                             DecisionTreeClassifier(random_state = 42, max_depth = 6))
        start = time.time()
        best_pipe_DT.fit(X_train, y_train)
        end = time.time()
        print(f'DT: {end - start} s')
        best_pipe_DT_PCA = make_pipeline(StandardScaler(),\
                             PCA(n_components = 0.95), \
                             DecisionTreeClassifier(random_state = 42, max_depth = 6))
        start = time.time()
        best_pipe_DT_PCA.fit(X_train, y_train)
        end = time.time()
        print(f'DT(pca): {end - start} s')
        # RF
        best_pipe_RF = make_pipeline(StandardScaler(),\
                             RandomForestClassifier(random_state = 42, max_depth = 6))
        start = time.time()
        best_pipe_RF.fit(X_train, y_train)
        end = time.time()
        print(f'RT: {end - start} s')
        best_pipe_RF_PCA = make_pipeline(StandardScaler(),\
                             PCA(n_components = 0.95), \
                             RandomForestClassifier(random_state = 42, max_depth = 6))
        start = time.time()
        best_pipe_RF_PCA.fit(X_train, y_train)
        end = time.time()
        print(f'RF(pca): {end - start} s')
        # Ada
        best_pipe_Ada = make_pipeline(StandardScaler(),\
                             AdaBoostClassifier(random_state = 42, n_estimators = 50))
        start = time.time()
        best_pipe_Ada.fit(X_train, y_train)
        end = time.time()
        print(f'Ada: {end - start} s')
        best_pipe_Ada_PCA = make_pipeline(StandardScaler(),\
                             PCA(n_components = 0.95), \
                             AdaBoostClassifier(random_state = 42, n_estimators = 50))
        start = time.time()
        best_pipe_Ada_PCA.fit(X_train, y_train)
        end = time.time()
        print(f'Ada(pca): {end - start} s')
        # knn
        best_pipe_knn = make_pipeline(StandardScaler(),\
                             KNeighborsClassifier(n_neighbors = 6))
        start = time.time()
        best_pipe_knn.fit(X_train, y_train)
        end = time.time()
        print(f'knn: {end - start} s')
        best_pipe_knn_PCA = make_pipeline(StandardScaler(),\
                             PCA(n_components = 0.95), \
                             KNeighborsClassifier(n_neighbors = 6))
        start = time.time()
        best_pipe_knn_PCA.fit(X_train, y_train)
        end = time.time()
        print(f'knn(pca): {end - start} s')
        # xgb
        best_pipe_xgb = make_pipeline(StandardScaler(),\
                             XGBClassifier(random_state = 42, max_depth = 2))
        start = time.time()
        best_pipe_xgb.fit(X_train, y_train)
        end = time.time()
        print(f'xgb: {end - start} s')
        best_pipe_xgb_PCA = make_pipeline(StandardScaler(),\
                             PCA(n_components = 0.95), \
                             XGBClassifier(random_state = 42, max_depth = 2))
        start = time.time()
        best_pipe_xgb_PCA.fit(X_train, y_train)
        end = time.time()
        print(f'xgb(pca): {end - start} s')
        # 繪圖
        fig, ax = plt.subplots()
        models = [
            ("Logistic Regression", best_pipe_LR),
            ("Decision Tree", best_pipe_DT),
            ("Random Forest", best_pipe_RF),
            ("AdaBoost", best_pipe_Ada),
            ("K-neighbors", best_pipe_knn),
            ("XGBoost", best_pipe_xgb),
            ("Logistic Regression (with PCA)", best_pipe_LR_PCA),
            ("Decision Tree (with PCA)", best_pipe_DT_PCA),
            ("Random Forest (with PCA)", best_pipe_RF_PCA),
            ("AdaBoost (with PCA)", best_pipe_Ada_PCA),
            ("K-neighbors", best_pipe_knn_PCA),
            ("XGBoost (with PCA)", best_pipe_xgb_PCA),
        ]
        model_displays = {}
        for name, pipeline in models:
            model_displays[name] = RocCurveDisplay.from_estimator(
                pipeline, X_test, y_test, ax=ax, name=name
            )
        _ = ax.set_title("ROC curve")
        plt.grid()
        plt.show()   
        
        
        
        # K-means之elbow
        distortions = []
        for i in range(1, 15):
            km = KMeans(n_clusters=i,init='k-means++',n_init=10,
            max_iter=300, random_state=0)
            km.fit(X_train_std)
            distortions.append(km.inertia_)
        plt.plot(range(1,15), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('K-menas elbow method')
        plt.grid()
        plt.show()
                
        
        
        # SVM
        from sklearn.metrics import pairwise_distances_argmin_min
        # 標準化數據
        std_scaler = StandardScaler()
        X_train_std = std_scaler.fit_transform(X_train)
        X_test_std = std_scaler.fit_transform(X_test)
        
        svc = SVC(random_state = 42, kernel = 'linear')
        km = KMeans(n_clusters=5).fit(X_train_std)
        closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X_train_std)
        svc.fit(X_train.iloc[closest, :], y_train.iloc[closest])
        svc_centers_predict = svc.predict(X_train.iloc[closest, :])
        
        svc_y_predict = []
        for km_predict_label in km.predict(X_test_std):
            svc_y_predict.append(svc_centers_predict[km_predict_label])
        svc_y_test_predict = np.array(svc_y_predict)
        print(classification_report(y_test, svc_y_test_predict, digits = 4))  
        fpr, tpr, _ = roc_curve(y_test, svc_y_test_predict, pos_label=1)
        print(f'AUC: {auc(fpr, tpr)}')
        
        
        
        
        
        
        
        
        
        
        
        
        
        # 檢驗
        best_pipe = make_pipeline(StandardScaler(),\
                              PCA(n_components = 0.95), \
                              SVC(random_state = 42, kernel = 'linear', probability = True))
        best_pipe.fit(X_train.iloc[closest, :], y_train.iloc[closest])
        # Check agreement test
        agreement_probs = best_pipe.predict_proba(check_agreement[features])[:,1]
        
        ks = evaluation.compute_ks(
            agreement_probs[check_agreement['signal'].values == 0],
            agreement_probs[check_agreement['signal'].values == 1],
            check_agreement[check_agreement['signal'] == 0]['weight'].values,
            check_agreement[check_agreement['signal'] == 1]['weight'].values)
        print('KS metric', ks, ks < 0.09)
        
        # Check correlation test
        correlation_probs = best_pipe.predict_proba(check_correlation[features])[:,1]
        
        cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
        print('CvM metric', cvm, cvm < 0.002)
        
        # Compute weighted AUC on the training data with min_ANNmuon > 0.4
        train_eval = train[train['min_ANNmuon'] > 0.4]
        train_probs = best_pipe.predict_proba(train_eval[features])[:,1]
        AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
        print('AUC', AUC)                    
        