import pickle

def predict_class(file_name, x_test, img_no, y_test):
    loaded_model = pickle.load(open(file_name, 'rb'))
    preds = loaded_model.predict(x_test, verbose=1)
    print ('Predicted Label: ',end='')
    if preds[img_no, 0]*100  >= 80:
        print ('Barren Land')
    elif preds[img_no, 1]*100 >= 80:
        print ('Forest Land')
    elif preds[img_no, 2]*100 >= 80:
        print ('Grassland')
    else:
        print ('Other')
        
    
    # Actual classification
    print ('Actual label: ',end='')
    if y_test[img_no, 0] == 1:
        print ('Barren Land')
    elif y_test[img_no, 1] == 1:
        print ('Forest Land')
    elif y_test[img_no, 2] == 1:
        print ('Grassland')
    else:
        print ('Other')
    
    return preds
