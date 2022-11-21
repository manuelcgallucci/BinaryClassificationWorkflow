

def main():
    
    BankNotes = {
        "data_path": ,
        "ftr_names": ,
        "out_path": ,
        "model": ,
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

        import_data(work["data_path"], work["ftr_names"], true_label_str=(np.Nan,np.Nan),normalize="std", class_label="last", dummy_cols=None, replace_nan="mean")


if __name__ == "__main__":
    main()