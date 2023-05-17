# 只有 "year" 作為特徵的模型的平均平方誤差 (MSE) 是 66.69，並且 R2 分數是 0.6126。
# 同時具有 "Year" 和 "IsModernCloser" 作為特徵的模型的 MSE 是 50.44，並且 R2 分數是 0.7110。
# 只有 "IsModernCloser" 作為特徵的模型的 MSE 是 57.17，並且 R2 分數是 0.6712。

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import glob
import os

csv_files = glob.glob('data/pitching_splits_closer_2008_2022/*.csv')

column_names = ['OPS', 'BAbip', 'tOPS+', 'BA', 'SLG', 'OBP']

save_df_list = []
non_save_df_list = []

for file in csv_files:
    df = pd.read_csv(file)

    # Extract player's name and year from the file name
    base = os.path.basename(file)
    file_info = os.path.splitext(base)[0].split('_')
    player_name = file_info[0]
    # ensure the year is an integer
    year = int(file_info[1])  

    # Add a new column 'IsModernCloser'
    if year <= 2016:
        is_modern_closer = False
    else:
        is_modern_closer = True

    # Filter the DataFrame based on the save situation condition and add player's name, year, and IsModernCloser
    save_situation_condition = df['Split'] == 'in Sv Situ'
    save_df = df.loc[save_situation_condition, column_names]
    save_df = save_df.assign(Player_Name=player_name, Year=year, IsModernCloser=is_modern_closer)
    save_df_list.append(save_df)

    # for non save situations
    non_save_situation_condition = df['Split'] == 'in non-Sv'
    non_save_df = df.loc[non_save_situation_condition, column_names]
    non_save_df = non_save_df.assign(Player_Name=player_name, Year=year, IsModernCloser=is_modern_closer)
    non_save_df_list.append(non_save_df)

# Concatenate all the DataFrames in the lists into single DataFrames
all_save_df = pd.concat(save_df_list)
all_non_save_df = pd.concat(non_save_df_list)

# Convert 'IsModernCloser' column to integer type (0 for False and 1 for True)
all_save_df['IsModernCloser'] = all_save_df['IsModernCloser'].astype(int)
all_non_save_df['IsModernCloser'] = all_non_save_df['IsModernCloser'].astype(int)

# Prepare for model training
features = column_names + ['Year', 'IsModernCloser']  
X = all_save_df[features]  
y = all_non_save_df[features]  

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print the performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE) with 'IsModernCloser': {mse}")
print(f"R-squared (R2 ) with 'IsModernCloser': {r2}")

