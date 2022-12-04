import workflow as wf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import os 

def main():
    
    BankNotes = {
        "name": "BankNotes",
        "data_path": "./data/data_banknote_authentication.txt",
        "ftr_names": ["variance of WTI", "skewness of WTI", "curtosis of WTI", "entropy"],
        "out_path": "./output/bank_notes/",
        "model": RandomForestClassifier(n_estimators=10, max_depth=6),
        "cross_validation": 10,
        "test_size": 0.3,
        "true_label": None,
        "dummy_cols": None,
    } 
    
    ChronicKidney = {
        "name": "ChronicKidneyDisease",
        "data_path": "./data/kidney_disease.csv",
        "ftr_names": ["id","age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","htn","dm","cad","appet","pe","ane"],
        "out_path": "./output/chronic_kidney",
        "model": RandomForestClassifier(n_estimators=10, max_depth=6),
        "cross_validation": 5,
        "test_size": 0.3,
        "true_label": ( "ckd", "notckd"),
        "dummy_cols": ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"], 
    }

    workflow = [BankNotes, ChronicKidney]

    for work in workflow:
        
        data, labels = wf.import_data(work["data_path"], work["ftr_names"], true_label_str=work["true_label"], dummy_cols=None)

        
        # Not interested in plotting the pairplot when the number of features is too big
        # It takes too long and it is not as useful 
        if len(work["ftr_names"]) <= 5:
            wf.plot_data(data, work["out_path"])
        metrics, _ = wf.train_model(work["model"], data, labels, work["test_size"], plot_results=work["out_path"], cross_validation=work["cross_validation"])
        
        # Print the results to the terminal
        print("\nResults for the dataset:", work["name"])
        if work["cross_validation"] is not None:
            print("Metrics averaged over {:d} cross validations:".format(work["cross_validation"]))
        else:
            print("Metrics:")
        wf.print_metrics(metrics)

if __name__ == "__main__":
    main()