import pandas as pd
import plotly.express as px
import pmdarima as pm
import datetime
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import csv

def plotChoropleth(data):
    fig = px.choropleth(data, locations='iso_code', color='mortality_rate', hover_name='location',
                    projection='natural earth', title='COVID Mortality Rates by Country', animation_frame='date')


    fig.show()

def useARIMA(csvData):
    # Format data for proper date/tiime for predictions
    csvData['date'] = pd.to_datetime(csvData['date'], format='%m/%d/%Y')
    csvData.sort_values(by=['date'], ascending=True)
    csvData.set_index('date', inplace=True)

    #Resample to fill in any missing values
    csvData = csvData.resample("D").ffill().reset_index()
    
    lastIndex = csvData.shape[0] - 1

    print("LAST INDEX - " + str(lastIndex))
    testIndex = round(lastIndex - (lastIndex * 0.2))
    print("HERE" + str(testIndex))
    training = csvData[:testIndex]
    print(training)
    testing = csvData[testIndex:]
    print(testing)

    numPredictions = 300

    df_diff = training.mortality_rate.diff()
    df_diff = df_diff.fillna(method='ffill')
    df_diff = df_diff.fillna(method='bfill')

    auto_arima = pm.arima.auto_arima(df_diff, stepwise=False, seasonal=False)
    test_auto = auto_arima.predict(n_periods=numPredictions)

    # Format ARIMA predictions into necessary rows
    isoCode = csvData['iso_code'].tail(1)
    isoCode = isoCode.tolist()[0]
    continent = csvData['continent'].tail(1)
    continent = continent.tolist()[0]
    location = csvData['location'].tail(1)
    location = location.tolist()[0]

    lastDate = csvData['date']

    # Formats the datetime object for use in visualization
    lastDate = pd.to_datetime(lastDate)  
    lastDate = lastDate.tolist()
    lastDate = pd.to_datetime(lastDate[lastIndex])
    
    # Inserts a new row for 
    lastValue = csvData['mortality_rate'].values[lastIndex]

    for arimaIndex in range(testIndex, numPredictions-1):
        newDate = lastDate + datetime.timedelta(days=arimaIndex)
        morRate = lastValue + test_auto.values[arimaIndex]
        newRow = np.asarray([newDate, isoCode, continent, location, 0, 0, 0, 0, morRate], dtype='object')
        csvData.loc[lastIndex + arimaIndex + 1] = newRow
        lastValue = morRate
        test_auto.values[arimaIndex] = morRate

    # Find and print out accuracy metrics
    with open('output3.csv', 'a', newline='') as output:
        writer = csv.writer(output, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        MAE = metrics.mean_absolute_error(testing.mortality_rate, test_auto[:len(testing)])
        MSE = metrics.mean_squared_error(testing.mortality_rate, test_auto[:len(testing)])
        RMSE = np.sqrt(metrics.mean_squared_error(testing.mortality_rate, test_auto[:len(testing)]))
        list = [location, MAE, MSE, RMSE]
        writer.writerow(list)

    return csvData

def main():
    testData = pd.read_csv('COVID.csv', header=0)
    countrylist = testData.location.unique()
    withresults = pd.DataFrame(columns=['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'mortality_rate'])

    for country in countrylist:
        tosend = testData.loc[testData["location"] == country]
        tosend = useARIMA(tosend)
        withresults = pd.concat([withresults, tosend])

    withresults.to_csv('withresults.csv', index=False)
    plotChoropleth(withresults)

if __name__=="__main__":
    main()