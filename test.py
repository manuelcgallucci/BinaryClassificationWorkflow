import unittest 
import pandas as pd
import numpy as np
import workflow as wf

class TestSum(unittest.TestCase):
    def test_import_data_replacements(self):
        replace_values = ["mean", "median", "remove"]
        for replace_value in replace_values:
            data, label = wf.import_data("./data/test/test_data.csv", ["dummy", "int", "float", "label"], true_label_str=None,normalize="std", class_label="last", dummy_cols=["dummy"], replace_nan=replace_value)
            
            correct_result = pd.read_csv("./data/test/test_result_"+replace_value+".csv")
            
            if replace_value != "remove":
                correct_label = np.array([1,1,1,1,1,0])
                self.assertEqual(data == correct_result, 0)
                self.assertEqual(label == correct_label, 0)
            else:
                correct_label = np.array([1,1,1,1,0])
                self.assertEqual(data == correct_result, 0)
                self.assertEqual(label == correct_label, 0)


if __name__ == '__main__':
    
    # unittest.main()
    data, label = wf.import_data("./data/test/test_data.csv", ["dummy", "int", "float", "label"], 
        true_label_str=None,normalize=None, class_label="last", dummy_cols=["dummy"], replace_nan="mode")

    print(label)
    print(data)