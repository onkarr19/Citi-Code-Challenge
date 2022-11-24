import pickle

from django.http import HttpResponse
from django.shortcuts import render

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import os
import glob

import matplotlib

matplotlib.use('Agg')

sns.set(style="whitegrid")

pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr

yf.pdr_override()

start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2020, 1, 1)

stockz = sorted(
    ["EEM", "SPY", "GDX", "XLF", "XOP", "AMLP", "FXI", "QQQ", "EWZ", "EFA", "USO", "HYG", "IAU", "IWM", "XLE", "XLU",
     "IEMG", "GDXJ", "SLV", "VWO", "XLP", "XLI", "OIH", "LQD", "XLK", "VEA", "TLT", "IEFA", "XLV", "EWJ", "GLD", "IYR",
     "BKLN", "EWH", "ASHR", "XLB", "RSX", "JNK", "KRE", "XBI", "AGG", "VNQ", "GOVT", "UNG", "IVV", "XLY", "EWT", "PFF",
     "XLRE", "MCHI", "INDA", "BND", "USMV", "EZU", "SMH", "XRT", "EWY", "IEF", "SPLV", "XLC", "IJR", "VIXY", "EWG",
     "EWW", "VTI", "VGK", "IBB", "PGX", "VOO", "EMB", "SCHF", "VEU", "SJNK", "EMLC", "XME", "DIA", "EWA", "VCSH",
     "JPST", "MLPA", "VCIT", "ITB", "ACWI", "KWEB", "EWC", "EWU", "BNDX", "SHY", "VT", "IWD", "VXUS", "MBB", "ACWX",
     "XHB", "BSV", "SHV", "FEZ", "IWF", "IGSB", "SPYV", "ITOT", "FPE", "FVD", "SHYG", "VYM", "BBJP", "DGRO", "KBE",
     "VTV", "SPAB", "SPIB", "IWR", "DBC", "BIL", "SPSB", "FLOT", "GLDM", "VIG", "XES", "SCHE", "TIP", "PDBC", "SPYG",
     "MINT", "SCZ", "SPDW", "PCY", "USHY", "IXUS", "NEAR", "EPI", "SPLG", "HYLB", "AAXJ", "SPEM", "VMBS", "BIV", "QUAL",
     "ILF", "EWP"])

df = pdr.get_data_yahoo(stockz, start, end)['Close']


def zscore(series):
    return (series - series.mean()) / np.std(series)


def visualize(a, b):
    ratios = df[a] / df[b]
    train_test_index = int(len(ratios) * .70)
    train = ratios[:train_test_index]
    test = ratios[train_test_index:]
    zscore(ratios).plot(figsize=(12, 6))
    plt.axhline(zscore(ratios).mean())
    plt.axhline(1.0, color='red')
    plt.axhline(-1.0, color='green')
    plt.xlim('2016-01-01', '2020-01-01')

    plt.savefig('pairTradingApp/static/img1.jpg')
    plt.close()

    ratios_mavg5 = train.rolling(window=5, center=False).mean()
    ratios_mavg60 = train.rolling(window=60, center=False).mean()
    std_60 = train.rolling(window=60, center=False).std()
    zscore_60_5 = (ratios_mavg5 - ratios_mavg60) / std_60
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train.values)
    plt.plot(ratios_mavg5.index, ratios_mavg5.values)
    plt.plot(ratios_mavg60.index, ratios_mavg60.values)
    plt.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])

    plt.ylabel('Ratio')
    plt.savefig('pairTradingApp/static/img2.jpg')
    plt.close()

    plt.figure(figsize=(12, 6))
    zscore_60_5.plot()
    plt.xlim('2016-03-25', '2018-07-01')
    plt.axhline(0, color='black')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.axhline(-1.0, color='green', linestyle='--')
    plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
    plt.savefig('pairTradingApp/static/img3.jpg')
    plt.close()

    plt.figure(figsize=(12, 6))

    train[160:].plot()
    buy = train.copy()
    sell = train.copy()
    buy[zscore_60_5 > -1] = 0
    sell[zscore_60_5 < 1] = 0
    buy[160:].plot(color='g', linestyle='None', marker='^')
    sell[160:].plot(color='r', linestyle='None', marker='^')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, ratios.min(), ratios.max()))
    plt.xlim('2016-08-15', '2018-07-07')
    plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
    plt.savefig('pairTradingApp/static/img4.jpg')
    plt.close()

    plt.figure(figsize=(12, 7))
    S1 = df[a].iloc[:train_test_index]
    S2 = df[b].iloc[:train_test_index]

    S1[60:].plot(color='b')
    S2[60:].plot(color='c')
    buyR = 0 * S1.copy()
    sellR = 0 * S1.copy()

    # When you buy the ratio, you buy stock S1 and sell S2
    buyR[buy != 0] = S1[buy != 0]
    sellR[buy != 0] = S2[buy != 0]

    # When you sell the ratio, you sell stock S1 and buy S2
    buyR[sell != 0] = S2[sell != 0]
    sellR[sell != 0] = S1[sell != 0]

    buyR[60:].plot(color='g', linestyle='None', marker='^')
    sellR[60:].plot(color='r', linestyle='None', marker='^')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))
    plt.ylim(5, 60)
    plt.xlim('2016-03-22', '2018-07-04')

    plt.legend([a, b, 'Buy Signal', 'Sell Signal'])
    plt.savefig('pairTradingApp/static/img5.jpg')
    plt.close()


def read_pickle():
    dbfile = open(r'pairTradingApp/map.pkl', 'rb')
    data = pickle.load(dbfile)
    dbfile.close()
    return data


data = read_pickle()
symbols = set()

for key in data:
    symbols.add(key[0])
    symbols.add(key[1])
symbols = list(sorted(symbols))


def index(request):
    # read_pickle()
    return render(request, 'index.html', {'symbol': symbols, 'data': data})


def test(request):
    if 'a' in request.GET and 'b' in request.GET:
        a = request.GET.get('a')
        b = request.GET.get('b')
        # print(data.get((a,b), -1))
        if a == b:
            res = "Please choose different instruments!"
        elif (a, b) in data:
            res = data.get((a, b))

            files = glob.glob('pairTradingApp/static/*')
            for f in files:
                try:
                    os.remove(f)
                except:
                    continue
            visualize(a, b)
        else:
            res = f"Couldn't find pair of {a} & {b}. Please select a different pair."
        return render(request, 'index.html', {'symbol': symbols, 'current': res, 'n': 5, 'a': a, 'b': b, 'check': True})
    return HttpResponse('get back to home')

# print(data[('VIXY', 'XES')])
