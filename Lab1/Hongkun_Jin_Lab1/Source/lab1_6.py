"""
5. Program a code which download a web page contains a table using Request library, then parse the
page using Beautiful soup library. You should save the information about the states and their capitals
in a file.

Sample input: https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States
Sample output: Save the table in this link into a file


"""
from bs4 import BeautifulSoup
import urllib.request
import os
import csv


def search_spider():

    url = "https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States"
    source_code = urllib.request.urlopen(url)
    plain_text = source_code
    soup = BeautifulSoup(plain_text, "html.parser")
    # parse source web to a variable
    table_states = soup.find('table', {'class': "wikitable sortable plainrowheaders"})

    # Top title of the table
    caption_1 = table_states.find('caption').getText()
    with open('csv_states.csv', 'a') as csvFile:
        csvFile.write(caption_1)
        csvFile.close()

    # The original Wiki page has the "thead" for showing the title of the table, but the text parsed by BeautifulSoup
    # did not has the "thead" . I will created the title by myself
    title_table = ["Name", "postal abbreviation", "Capitals", "Largest City", "Established", "Population",
                   "Total Area(m2)", "Total Area(km2)", "Land Area(m2)", "Land Area(km2)", "Water Area(m2)",
                   "Water Area(km2)", "Number of Reps."]
    with open('csv_states.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(title_table)
        csvFile.close()

    table_tbody = table_states.find('tbody')
    tbody_tr = table_tbody.find_all('tr')
    all_1 = []
    for tr in tbody_tr:
        row_all = []
        th_tr = tr.find_all('th')
        for th in th_tr:
            a_th = th.find_all('a')
            for a in a_th:
                if a.get('title'):
                    row_all.append(a.get('title'))

        td_tr = tr.find_all('td')
        for td in td_tr:
            if td.get('colspan') == '2':
                row_all.append(td.getText())
            row_all.append(td.getText())

        all_1.append(row_all)

        with open('csv_states.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row_all)


search = input('type "s" to start wiliScrap, type "q" to exit')

if not os.path.exists(search):
    print("Creating .csv file ...")
    open('csv_states.csv', 'a+', encoding='utf-8')

if search == 'q' or search == 'Q':
    exit()
else:
    search_spider()
