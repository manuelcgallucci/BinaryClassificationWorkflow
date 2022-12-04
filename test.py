import unittest 
import pandas as pd
import numpy as np
import workflow as wf

class TestSum(unittest.TestCase):
    def test_import_data_replacements(self):
        replace_values = [True, False]
        for replace_value in replace_values:
            data, label = wf.import_data("./data/test/test_data.csv", ["dummy", "int", "float", "label"], true_label_str=None,normalize=None, class_label="last", dummy_cols=["dummy"], replace_nan=replace_value)
            
            correct_result = pd.read_csv("./data/test/test_result_"+str(replace_value)+".csv")
            
            if replace_value:
                correct_label = np.array([1,1,1,1,1,0])
                dataBool = data.reset_index(drop=True) == correct_result.reset_index(drop=True)
                self.assertEqual(dataBool.all().all(), True)
                self.assertEqual(np.array_equal(label, correct_label), True)
            else:
                correct_label = np.array([1,1,1,1,0])
                dataBool = data.reset_index(drop=True) == correct_result.reset_index(drop=True)
                self.assertEqual(dataBool.all().all(), True)
                self.assertEqual(np.array_equal(label, correct_label), True)


if __name__ == '__main__':
    
    unittest.main()
    