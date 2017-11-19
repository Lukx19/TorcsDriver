import pandas as pd 
from sklearn import preprocessing

df_list = ["aalborg_provided.csv", "f-speedway_provided.csv", "Forza_1_41_50.csv", "Corkscrew_01_26_01.csv"]
for i in range(len(df_list)):
    print(df_list[i])
    df = pd.read_csv(df_list[i], index_col = False)
    
    colnames = df.columns.values 
    # print(colnames)
    print("max Brake: ", max(df["BRAKE"]))
    print("min Brake: ", min(df["BRAKE"]))
    print()
    print("max Steering: ", max(df["STEERING"]))
    print("min Steering: ", min(df["STEERING"]))
    print()
    print("max Speed: ", max(df["SPEED"]))
    print("min Speed: ", min(df["SPEED"]))
    print("-----")
    print()
    df.loc[df["BRAKE"] >0, "BRAKE"] = 1
    # print(df.head())

    min_max_scaler = preprocessing.MinMaxScaler()
    df_minmax = pd.DataFrame(min_max_scaler.fit_transform(df))
    df_minmax.columns = colnames

    filename = "norm_" + df_list[i]

    df_minmax.to_csv(filename, index = False)
