import pandas as pd
df = pd.read_csv('TGN-Nature/实验代码/fig3/mqar_training_curves.csv')
print(df['model'].unique())
