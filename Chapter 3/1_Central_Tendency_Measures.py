# Importing the required library
import pandas_datareader as pdr

# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date   = '2023-01-23'

# Creating a dataframe and downloading the CPI data
# CPIAUCSL은 FRED에서 정의한 시계열 ID
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Printing the latest five observations of the dataframe
print(cpi.tail())

# Checking if there are nan values in the CPI dataframe
count_nan = cpi['CPIAUCSL'].isnull().sum()

# Printing the result
print('Number of nan values in the CPI dataframe: ' + str(count_nan))

# Transforming the CPI into a year-on-year measure
# CPI를 연도 대비 변화율(%)로 변환
# pct_change(periods=12)는 1년 전 대비 몇 % 상승했는지를 계산
cpi = cpi.pct_change(periods = 12, axis = 0) * 100

# Dropping the nan values from the rows
# nan 값 제거
cpi = cpi.dropna()

# Calculating the mean of the CPI over the last 20 years
# 최근 240개월(20년) 데이터만 추출해서 평균 계산
# 2003 - 2023
cpi_latest = cpi.iloc[-240:]
mean = cpi_latest["CPIAUCSL"].mean()

# Printing the result
print('The mean of the dataset: ' + str(mean), '%')

# Importing the required library
import matplotlib.pyplot as plt

# Plotting the latest observations in black with a label
plt.plot(cpi_latest[:], color = 'black', linewidth = 1.5, 
         label = 'Change in CPI Year-on-Year')

# Plotting horizontal lines that represent the mean and the zero threshold
plt.axhline(y = mean, color = 'red', linestyle = 'dashed', 
         label = 'Mean')
plt.axhline(y = 0, color = 'blue', linestyle = 'dashed', linewidth = 1)

# Calling a grid to facilitate the visual component
plt.grid()

# Calling the legend function so it appears with the chart
plt.legend()
plt.close()

# Calculating the median of the dataset
median = cpi_latest["CPIAUCSL"].median() 

# Printing the result
print('The median of the dataset: ' + str(median), '%')

# Plotting the latest observations in black with a label
plt.plot(cpi_latest[:], color = 'black', linewidth = 1.5, 
         label = 'Change in CPI Year-on-Year')

# Plotting horizontal lines that represent the median and the zero threshold
plt.axhline(y = median, color = 'red', linestyle = 'dashed', 
            label = 'Median')
plt.axhline(y = 0, color = 'blue', linestyle = 'dashed', linewidth = 1)

# Calling a grid to facilitate the visual component
plt.grid()

# Calling the legend function so it appears with the chart
plt.legend()

input("Press Enter to exit")  # 종료 전 입력 대기










