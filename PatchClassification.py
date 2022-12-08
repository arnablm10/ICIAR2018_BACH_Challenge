names = [
        "K Nearest Neighbour Classifier",
        'SVM',
        "Random Forest Classifier",
        "AdaBoost Classifier", 
        "XGB Classifier",
         ]
classifiers = [
    KNeighborsClassifier(),
    SVC(probability=True, kernel='rbf', degree = 2),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    XGBClassifier(),
        ]

zipped_clf = zip(names,classifiers)

def classifier_summary(pipeline, X_train, y_train, X_val, y_val):
    sentiment_fit = pipeline.fit(X_train, y_train)
    
    y_pred_train= sentiment_fit.predict(X_train)
    y_pred_val = sentiment_fit.predict(X_val)
    tmp1 = sentiment_fit.predict_proba(X_train)
    tmp2 = sentiment_fit.predict_proba(X_val)
    train_accuracy = np.round(accuracy_score(y_train, y_pred_train),4)*100    
    val_accuracy = np.round(accuracy_score(y_val, y_pred_val),4)*100
    print(train_accuracy, " " , val_accuracy)
    return tmp1, tmp2
    
def classifier_comparator(X_train,y_train,X_val,y_val, classifier=zipped_clf): 
    train_prob = []
    test_prob = []
    ensemble_techs = [max_ensemble, product_ensemble, average_ensemble]
    for n,c in classifier:
        print("hi")
        checker_pipeline = Pipeline([('classifier', c)])
        print("Fitting {} on features ".format(n))
        #print(c)
        a, b = classifier_summary(checker_pipeline,X_train, y_train, X_val, y_val)
        train_prob.append(a)
        test_prob.append(b)
    
    train_output = [ensemble_tech(train_prob) for ensemble_tech in ensemble_techs]
    test_output = [ensemble_tech(test_prob) for ensemble_tech in ensemble_techs]
    
    return train_output, test_output
