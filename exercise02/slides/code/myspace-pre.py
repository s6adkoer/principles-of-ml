import pandas as pd
# Loading data as pandas Dataframe
df = pd.read_csv("myspace.csv", 
                 header=None, 
                 names=["Week", "Accesses"])
# Filter leading zeros only with cumulative sum
df = df[np.cumsum(df.Accesses) > 0]
h = np.array(df.Accesses, dtype=float)
t = np.arange(1, len(h)+1, dtype=float)