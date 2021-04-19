# import MetaTrader5 as mt5
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # df = pd.read_csv('WIN$NM15_data.csv', sep=',')

# # print(df)

# # periodo = 21
# # desvios = 2

# # df["Desvio"] = df["Close"].rolling(periodo).std()
# # df["MM"] = df["Close"].rolling(periodo).mean()
# # df["Banda_Sup"] = df["MM"] + (df["Desvio"]*desvios)
# # df["Banda_Inf"] = df["MM"] - (df["Desvio"]*desvios)

# # df = df.dropna(axis=0)

# # df[["Close", "MM", "Banda_Sup", "Banda_Inf"]][:1000].plot(grid=True, figsize=(20, 15), linewidth=1, fontsize=15, color=["darkblue", "orange", "green", "red"])
# # plt.xlabel("Data", fontsize=15)
# # plt.ylabel("Pontos", fontsize=15)
# # plt.title(symbol, fontsize=25)
# # plt.legend()

# # # Construcao dos alvos
# # periodos = 5

# # # Alvo - Retorno
# # df.loc[:, "Retorno"] = df["Close"].pct_change(periodos)
# # df.loc[:, "Alvo"] = df["Retorno"].shift(-periodos)

# # df = df.dropna(axis=0) 

# # df.loc[:, "Regra"] = np.where(df.loc[:, "Close"] > df.loc[:, "Banda_Sup"], 1, 0)
# # df.loc[:, "Regra"] = np.where(df.loc[:, "Close"] < df.loc[:, "Banda_Inf"], -1, df.loc[: , "Regra"])

# # df.loc[:, "Trade"] = df.loc[:, "Regra"]*df.loc[:, "Alvo"]

# # df.loc[:, "Retorno_Trade_BB"] = df["Trade"].cumsum()

# # df["Retorno_Trade_BB"].plot(figsize=(20, 15), linewidth = 3, fontsize=15, color="green")
# # plt.xlabel("Data", fontsize=15);
# # plt.ylabel("Pontos", fontsize=15);
# # plt.title(symbol, fontsize=25)
# # plt.legend()

import MetaTrader5 as mt5


mt5.initialize()

lote = 100

symbol = 'WINM21'

mt5.symbol_select(symbol, True)

price = mt5.symbol_info_tick(symbol).last

dev = 2

request = {
     'action': mt5.TRADE_ACTION_DEAL,
     'symbol': symbol,
     'volume': float(lote),
     'type': mt5.ORDER_TYPE_SELL,
     'price': price,
     'sl': price + 100.0,
     'tp': price - 300.0,
     'deviation': dev,
     'magic': 123,
     'comment': 'Info',
     'type_time': mt5.ORDER_TIME_GTC,
     'type_filling': mt5.ORDER_FILLING_RETURN
   }


result = mt5.order_send(request)

print(result)

# # print(price)

# mt5.shutdown()

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

from sklearn import tree

my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(predictions, y_test))
