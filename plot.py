import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#read data from csv
data = pd.read_csv('CSVs/FinalResult2.csv', usecols=['created_at','score'], parse_dates=['created_at'])
#set date as index
data.set_index('created_at',inplace=True)

#plot data
fig, ax = plt.subplots(figsize=(30,10))
data.plot(kind='bar', ax=ax)

#set ticks every week
ax.xaxis.set_major_locator(mdates.MonthLocator())
#set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.savefig("/Users/slytanix/Desktop/result-new.png")
#plt.show()