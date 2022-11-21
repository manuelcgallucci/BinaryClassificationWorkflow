import workflow as wf
from sklearn.ensemble import RandomForestClassifier

def main():
    
    BankNotes = {
        "data_path": "./data/data_banknote_authentication.txt",
        "ftr_names": ,
        "out_path": "./output/bank_notes",
        "model": RandomForestClassifier(n_estimators=10, max_depth=6),
        "cross_validation": 5,
        "test_size": 0.3,
        "true_label": ( , ),
        "dummy_cols": None,
    }
    
    ChronicKidney = {
        "data_path": "./data/kidney_disease.csv",
        "ftr_names": [""],
        "out_path": "./output/chronic_kidney",
        "model": RandomForestClassifier(n_estimators=10, max_depth=6),
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