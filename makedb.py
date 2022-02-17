import csv
import json


# Function to convert a CSV to JSON
# Takes the file paths as arguments
import pandas as pd


def make_json(csvFilePath, jsonFilePath):

    # create a dictionary
    data = {}
    num = 0

    data = pd.read_csv(csvFilePath, usecols=['ALink','SName', 'Lyric', 'Idiom'], dtype={
        "ALink": str,
        "SName": str,
        "Lyric": str,
        "Idiom": str
    })

    data['data'] = data.apply(
        lambda x: {'artist': x.ALink.replace("/",""),
                   'song' : x.SName,
                   'lyric' : x.Lyric,
                   'idiom' : x.Idiom},
        axis=1)

    s = data.agg(lambda x: list(x))['data']
    results = []
    for index, subset in s.iteritems():
        if subset['idiom'] == "ENGLISH":
            num = num + 1
            results.append({'index': num, 'data': subset})

    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(results, indent=4))

    # # Open a csv reader called DictReader
    # with open(csvFilePath, encoding='utf-8') as csvf:
    #     csvReader = csv.DictReader(csvf)
    #
    #     # Convert each row into a dictionary
    #     # and add it to data
    #     for rows in csvReader:
    #
    #         # Assuming a column named 'No' to
    #         # be the primary key
    #         if data[]
    #             key = num
    #             data[key] = rows
    #             num = num+1
    #
    # # Open a json writer, and use the json.dumps()
    # # function to dump data
    # with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
    #     jsonf.write(json.dumps(data, indent=4))