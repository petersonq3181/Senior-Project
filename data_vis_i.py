import pandas as pd 
import matplotlib.pyplot as plt

# ----- 
# to inspect distributsion etc. of the data befor preprocessing 


fn = 'MorroBayHeights.csv'

csvFile = pd.read_csv(fn)


X_name = 'lotusSigh_mt'

Y_name = 'lotusMaxBWH_ft'
# make sure to standardize units to meters 
csvFile[Y_name] = csvFile[Y_name] * 0.3048 

# # mean centering 
# csvFile[X_name] = csvFile[X_name] - csvFile[X_name].mean()
# csvFile[Y_name] = csvFile[Y_name] - csvFile[Y_name].mean()

# # scaling by standard deviation 
# csvFile[X_name] = csvFile[X_name] / csvFile[X_name].std()
# csvFile[Y_name] = csvFile[Y_name] / csvFile[Y_name].std()

# plotting 
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(csvFile[X_name], bins=30, alpha=0.7)
plt.title('lotusSigh')
plt.xlabel('lotusSigh (m)')
plt.ylabel('Frequency')
plt.xlim([0, 7])

plt.subplot(1, 2, 2)
plt.hist(csvFile[Y_name], bins=30, alpha=0.7)
plt.title('lotusMaxBWH')
plt.xlabel('lotusMaxBWH (m)')
plt.ylabel('Frequency')
plt.xlim([0, 7])

plt.savefig('/Users/Quinn/Downloads/fig.png')

plt.tight_layout()
plt.show()
