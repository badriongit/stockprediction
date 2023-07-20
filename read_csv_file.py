import pandas as pd
df = pd.read_csv ('file_name.csv')
print(df)

work = df[["Date","Prev Close","Open Price","High Price","Low Price","Close Price"]]

work['O-C'] = work['Open Price'] - work['Close Price']
work['H-L'] = work['High Price'] - work['Low Price']

