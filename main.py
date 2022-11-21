import workflow as wf
from sklearn.ensemble import RandomForestClassifier

def main():
    
    BankNotes = {
        "data_path": ,
        "ftr_names": ,
        "out_path": "./",
        "model": RandomForestClassifier(n_estimators=10, max_depth=6),
        "cross_validation": 5,
        "test_size": 0.3,
        "true_label": ( , ),
    }
    
    ChronicKidney = {
        "data_path": ,
        "col_names": ,
        "ftr_path": ,
        "model": ,
        "cross_validation": 5,
        "test_size": 0.3,
        "true_label": ( , ),
        "dummy_cols": ,
    }

    workflow = [BankNotes, ChronicKidney]

    for work in workflow:

        data, labels = wf.import_data(work["data_path"], work["ftr_names"], true_label_str=work["true_label"], dummy_cols=None)
        

        break
if __name__ == "__main__":
    main()