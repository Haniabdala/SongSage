import bs4
import lxml
import pandas as pd
import urllib
import re
import datetime
from urllib import request

def fetch_billboard_data_for_week(date_str):
    url = f"https://www.billboard.com/charts/hot-100/{date_str}/"
    request_text = request.urlopen(url).read()
    page = bs4.BeautifulSoup(request_text, "lxml")
    
    table = page.find('div', class_="pmc-paywall")
    
    titles = table.find_all(lambda tag: tag.name == "h3" and tag.get("class") and "c-title" in tag["class"] and "a-no-trucate" in tag["class"])
    
    artists = table.find_all(lambda tag: tag.name == "span" and tag.get("class") and "c-label" in tag["class"] and "a-no-trucate" in tag["class"])
    
    songs = [(title.get_text(strip=True), artist.get_text(strip=True)) for title, artist in zip(titles, artists)]
    
    return songs

song_retr = []
artist_retr = []
year_retr = []

year = 2010 

while year < 2025:
    for week_count in range(1, 53):
        start_date = datetime.date(year, 1, 1)
        current_date = start_date + datetime.timedelta(weeks=week_count - 1)
        chart_date_str = current_date.strftime("%Y-%m-%d")
        
        print(f"Fetching data for {chart_date_str}")
        
        try:
            songs = fetch_billboard_data_for_week(chart_date_str)
        
            for title, artist in songs:
                    song_retr.append(title)
                    artist_retr.append(artist)
                    year_retr.append(year)
        except Exception as e:
            print(f"Error fetching data for {chart_date_str}: {e}")
    year+=1



data = {
    'Year': year_retr,
    'Song Titles': song_retr,
    'Artist': artist_retr
}
df = pd.DataFrame(data)

df_no_duplicates = df.drop_duplicates()
df_no_duplicates.to_csv('songs_data.csv', index=False)  
