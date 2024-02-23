import wfdb
import numpy as np
import ast
import pandas as pd
from torch.utils.data import Dataset


num2class = np.array(['NORM', 'MI', 'STTC', 'CD', 'HYP'])


class CustomDataset(Dataset):
    def __init__(self, text, ecg, labels):
        self.text = text
        self.ecg = ecg
        self.labels = labels
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        sample = {
            'ecg_input': self.ecg[idx],
            'text_input': self.text[idx],
            'label': self.labels[idx]
        }
        return sample
        
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data
    
def load_data():
    path = "/AIRvePFS/dair/users/hongcd/home/cad/data/ptb-xl/"
    # path = 'path'
    sampling_rate=500

# load and convert annotation data
    Y = pd.read_csv(path+'updated_ptb.csv', index_col='ecg_id') 
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    ecg_input = np.transpose(X, (0, 2, 1))
    reports = Y.translated_report.values.tolist() ### 翻译好的描述
    
    code2desc = {
    'NORM': 'Normal ECG',
    'STTC': 'ST/T change',
    'HYP': 'Hypertrophy',
    'MI': 'Myocardial Infarction',
    'CD': 'Conduction Disturbance'
    }
    
    labels = [] 
    for code in Y['diagnostic_superclass']:
        if code:
            scp = code[0]
            if scp in code2desc:
                labels.append(scp)   # 取第一个label
            else:
                labels.append('')    # 如果没有diagnostic label就空
        else:
            labels.append('')

    y = 
  
    dataset = CustomDataset(text_data = reports, ecg_data = ecg_input, labels = labels)
    train_ratio = 0.8
    val_ratio = 0.1
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    return train_set, val_set, test_set



if __name__ == "__main__":
    train_set, val_set, test_set = load_data()