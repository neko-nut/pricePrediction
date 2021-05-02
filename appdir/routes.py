from flask import request, jsonify

from appdir import application
import pandas as pd

from appdir.config import Config
from sklearn.neighbors import KNeighborsClassifier


def init():
    attributes = [
        "Date of Sale (dd/mm/yyyy)",
        "Address",
        "Postal Code",
        "County",
        "Price",
        "Not Full Market Price",
        "VAT Exclusive",
        "Description of Property",
        "Property Size Description"
    ]

    df = pd.read_csv(Config.Data, encoding='unicode_escape')
    df.columns = attributes
    df = df.drop(["Postal Code"], axis=1)
    df.dropna(axis=0, how='any', inplace=True)
    for i in df.index:
        df.loc[i, "Month"] = str(df["Date of Sale (dd/mm/yyyy)"][i][3:5])
        df.loc[i, "Year"] = str(df["Date of Sale (dd/mm/yyyy)"][i][6:10])
        if df.loc[i, "Price"] < 50000:
            df.loc[i, "Price"] = 0
        elif df.loc[i, "Price"] < 100000:
            df.loc[i, "Price"] = 1
        elif df.loc[i, "Price"] < 150000:
            df.loc[i, "Price"] = 2
        elif df.loc[i, "Price"] < 200000:
            df.loc[i, "Price"] = 3
        elif df.loc[i, "Price"] < 250000:
            df.loc[i, "Price"] = 4
        elif df.loc[i, "Price"] < 300000:
            df.loc[i, "Price"] = 5
        elif df.loc[i, "Price"] < 350000:
            df.loc[i, "Price"] = 6
        elif df.loc[i, "Price"] < 400000:
            df.loc[i, "Price"] = 7
        elif df.loc[i, "Price"] < 450000:
            df.loc[i, "Price"] = 8
        else:
            df.loc[i, "Price"] = 9

    num_encode = {
        "County": {
            "Dublin": 0,
            "Cork": 1,
            "Kildare": 2,
            "Meath": 3,
            "Galway": 4,
            "Wicklow": 5,
            "Louth": 6,
            "Wexford": 7,
            "Limerick": 8,
            "Kerry": 9,
            "Donegal": 10,
            "Cavan": 11,
            "Laois": 12,
            "Waterford": 13,
            "Clare": 14,
            "Mayo": 15,
            "Tipperary": 16,
            "Sligo": 17,
            "Westmeath": 18,
            "Roscommon": 19,
            "Kilkenny": 20,
            "Carlow": 21,
            "Leitrim": 22,
            "Monaghan": 23,
            "Longford": 24,
            "Offaly": 25
        },
        "Not Full Market Price": {
            "No": 0,
            "Yes": 1
        },
        "VAT Exclusive": {
            "No": 0,
            "Yes": 1
        },
        "Description of Property": {
            "New Dwelling house /Apartment": 0,
            "Second-Hand Dwelling house /Apartment": 1
        },
        "Property Size Description": {
            "less than 38 sq metres": 0,
            "greater than or equal to 38 sq metres and less than 125 sq metres": 1,
            "greater than 125 sq metres": 2,
            "greater than or equal to 125 sq metres": 2,
        }
    }
    df.replace(num_encode, inplace=True)
    global describe
    describe = [
        "Year",
        "Month",
        "County",
        "Not Full Market Price",
        "VAT Exclusive",
        "Property Size Description"
    ]
    x = df[describe]
    y = df["Price"]
    global knn
    knn = KNeighborsClassifier(n_neighbors=116, weights="distance")
    knn.fit(x, y)
    return "success"


@application.route('/predict', methods=['GET', 'POST'])
def predict():
    # test data (used in GET only)
    year = 2021
    month = 1
    county = 0
    full = 0
    pro = 0
    vat = 0
    size = 1

    # user input
    if request.method == "POST":
        year = request.form.get('year')
        month = request.form.get('month')
        county = request.form.get('county')
        full = request.form.get('full')
        pro = request.form.get('property')
        vat = request.form.get('vat')
        size = int(request.form.get('size'))

    if size < 35:
        size = 0
    elif size < 125:
        size = 1
    else:
        size = 2

    df_empty = pd.DataFrame([[year, month, county, full, vat, size]], columns=describe)
    res = knn.predict(df_empty)

    if pro == 1:
        if res[0] > 0:
            res = res[0] - 1
    else:
        res = res[0]

    # init except values
    max = 2147483647
    min = 450000

    res_map = {
        0: (0, 50000),
        1: (50000, 100000),
        2: (100000, 150000),
        3: (150000, 200000),
        4: (200000, 250000),
        5: (250000, 300000),
        6: (300000, 350000),
        7: (350000, 400000),
        8: (400000, 450000),
    }

    if res in res_map.keys():
        min, max = res_map[res]

    result = {
        "code": 200,
        "msg": "OK",
        "data": {
            "min": min,
            "max": max
        }
    }
    return jsonify(result)


# auto init
init()
