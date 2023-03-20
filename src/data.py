import os
from .config import config
import pandas as pd
from sklearn.model_selection import train_test_split
class data:
    def data(path):
        X,Y= [],[]
        dirs= sorted(os.listdir(path))
        for dir_ in dirs:
            for file in os.listdir(path+dir_):
    #             label= np.zeros(len(dirs))
                label= dirs.index(dir_)
                Y.append(label)
                
                X.append(dir_+'/'+file)
        return pd.DataFrame({'image_id': X,'label':Y},index= np.arange(0,len(X)))

    train= data(config.train_path)
    test= data(config.test_path)

    train_df, valid_df= train_test_split(train,test_size=0.20,shuffle= True, random_state= 42,stratify= train.label.values )