# Nick Chandler
# 17.01.2025
# Web scraper for csv weather time series data
# Website: https://wx1.slackology.net/data
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time



def isConvertibleToFloat(s: str) -> bool:
    '''
    Determines if a string s is convertible to a float
    Input:
        s - string
    Output:
        bool
    '''
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def collectOneDay(url: str):
    '''
    Collects the data of one day
    Input:
        Url of the form: "https://wx1.slackology.net/data/YYYY/bme680.dat.YYYYMMDD"
    Output:
        Pandas Dataframe with columns:
            Datetime, Temp (C), Humidity (%), Pressure (kPa), AirQ (Ohms)
    '''
    # url = "https://wx1.slackology.net/data/2018/bme680.dat.20180923"  # Testing url
    
    # 1 - fetch webpage content
    response = requests.get(url)
    response.raise_for_status()  # Raise an error upon failed request
    
    # 2 - parse the lines
    lines = response.text.strip().split('\n')
    
    # 3 - extract the data fields
    data = []
    extract_value = lambda s: s.split(':')[1].strip().split()[0]  # Deal with the formatting, get the desired value
    clean_data = lambda s: ''.join(c for c in s if c.isprintable())
    
    for line in lines:
        parts = line.split('\t')
        
        iso_time = clean_data(parts[0].strip())
        
        temp_string = extract_value(parts[1])
        temp = float(temp_string) if isConvertibleToFloat(temp_string) else float('nan')
        
        humidity_string = extract_value(parts[2])
        humidity = float(humidity_string) if isConvertibleToFloat(humidity_string) else float('nan')
        
        pressure_string = extract_value(parts[3])
        pressure = float(pressure_string) if isConvertibleToFloat(pressure_string) else float('nan')
        
        air_quality_string = extract_value(parts[4])
        air_quality = float(air_quality_string) if isConvertibleToFloat(air_quality_string) else float('nan')
        
        data.append([iso_time, temp, humidity, pressure, air_quality])

    # 4 - create dataframe
    df = pd.DataFrame(data, columns=['ISO Time', 'Temperature (C)', 'Humidity (%)', 'Pressure (kPa)', 'Air Quality (Ohms)'])

    # 5 - convert iso time to datetime
    df['ISO Time'] = pd.to_datetime(df['ISO Time'], format='%Y%m%d%H%M%S')
    return df


def obtainFilenames(url: str) -> [str]:
    '''
    Obtains the filenames starting with "bme680.dat." from each of the specified directories
    Input: 
        Url of the year directory
    Output: 
        list of filename strings to be concatenated of the form: "bme680.dat.YYYYMMDD"
    '''
    # url = "https://wx1.slackology.net/data/2018/"  # For debugging purposes
    
    # 1 - fetch the webpage
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request fails
    html_content = response.text

    # 2 - parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 3 - locate the tbody
    table_body = soup.find('tbody')
    
    # 4 - extract the rows and link text
    data = []
    for row in table_body.find_all('tr'):
        if 'dir' in row.get('class', []):
            continue
        
        # Extract the first <td> element
        first_td = row.find('td')
        if first_td:
            text = first_td.text.strip()  # The text from the first <td> ele't
            
            # Save only the bme files
            if text.startswith("bme680"):
                data.append(text)  
    return data


def main():
    # Modify this to change which year directories are queried
    years = ["2025"]  # ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
    base_url = "https://wx1.slackology.net/data/"

    for year in years:
        # Status update
        print("Working on year: ", year)
        
        # Prepare URL for writing to csv
        writing_path = "data/" + year + "_BME680SensorData.csv"
        
        # Prepare URl for web scraping
        year_url = base_url + year + "/"  # https://wx1.slackology.net/data/YYYY/
        
        # Obtain a list containing "bme680.dat.YYYYMMDD" strings
        day_filenames = obtainFilenames(year_url)
        
        # Collect the first day's data and write it to a new csv
        day_url = year_url + day_filenames[0]
        day_dataframe = collectOneDay(day_url)
        day_dataframe.to_csv(writing_path, mode='w', header=True)
        
        # Iterate through each day, skipping the first
        for day in day_filenames[1:]:
            # Prepare the url for each day
            day_url = year_url + day  # https://wx1.slackology.net/data/YYYY/bme680.dat.YYYYMMDD
            
            # obtaining a pandas dataframe of TS data for each day
            day_dataframe = collectOneDay(day_url)
            
            # append the day's data to the end of the csv for year
            day_dataframe.to_csv(writing_path, mode='a', header=False)
            
            # Status update
            print("Finished: ", day)

if __name__ == "__main__":
    main()
    