import sys
sys.path.insert(1, '/home/pi/dev/')
sys.path.insert(1, '/home/pi/dev/ssh')

import yfinance as yf
import pandas as pd
import sys, pprint, math, csv
import scp
from scp_keys import upload_ftp_location
from datetime import datetime


def readStockTickers(exchange):
	tickersDf = pd.read_csv('../data/reference/' + exchange + '.csv') 
	print(tickersDf)
	return tickersDf['ticker'].values.tolist()

"""**Download stock data from Yahoo Finance**"""
def downloadUSStockTickers():
	## Download NYSE, NASDAQ, BATS, IEXG tickers
	# Download latest stock ticker list from ftp.nasdaqtrader.com
	from ftplib import FTP

	directory = 'symboldirectory'
	filenames = ('otherlisted.txt', 'nasdaqlisted.txt')

	ftp = FTP('ftp.nasdaqtrader.com')
	ftp.login()
	ftp.cwd(directory)

	for item in filenames:
	    ftp.retrbinary('RETR {0}'.format(item), open(item, 'wb').write)

	ftp.quit()

	# Create dataframes from the nasdaqlisted and otherlisted FTP files.
	nasdaqDf = pd.read_csv('nasdaqlisted.txt', '|')
	otherDf = pd.read_csv('otherlisted.txt', '|')
	nasdaqDf.drop(nasdaqDf.tail(1).index,inplace=True) # drop last row
	otherDf.drop(otherDf.tail(1).index,inplace=True) # drop last row

	# extract relevant stocks into nyseDf, arcaDF, then into nyseTickers, arcaTickers, etc
	nyseDf = otherDf.loc[(otherDf['Exchange'] == 'A') | (otherDf['Exchange'] == 'N')]
	arcaDf = otherDf.loc[(otherDf['Exchange'] == 'P')]
	batsDf = otherDf.loc[(otherDf['Exchange'] == 'Z')]
	iexgDf = otherDf.loc[(otherDf['Exchange'] == 'V')]

	nasdaqTickers = nasdaqDf['Symbol'].tolist()
	nyseTickers = nyseDf['NASDAQ Symbol'].tolist()
	arcaTickers = arcaDf['NASDAQ Symbol'].tolist()
	batsTickers = batsDf['NASDAQ Symbol'].tolist()
	iexgTickers = iexgDf['NASDAQ Symbol'].tolist()

	## import list of SPY stocks
	spyTickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
	spyTickers = spyTickers[0]['Symbol']
	spyTickers = list(spyTickers)
	#print(spyTickers)

	with open("spy500.csv", "w", newline="") as csvFile:
	    writer = csv.writer(csvFile)
	    writer.writerow('ticker')
	    for val in spyTickers:
	        writer.writerow([val])

def importHKSEStockTickers():
	'''## import list of HKSE
	# Read the CSV into a pandas data frame (df)
	#   With a df you can do many things
	#   most important: visualize data with Seaborn'''
	hkseDf = pd.read_csv('drive/My Drive/Colab Notebooks/hkse.csv', delimiter=',')
	hkseTickers = hkseDf['symbol'].to_list()

# import progressbar (from here: https://stackoverflow.com/a/34482761)
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        # file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n")
    #file.flush()

def downloadYFinanceData(tickers, rfRateAnnualized, rfRate, period):
	#Example tickers:
	#tickers = ["SPY", "AAPL", "BCE"]
	#tickers = spyTickers
	#tickers = hkseTickers
	#tickers = ['NVDA','GOOGL']
	#tickers = arcaTickers
	#tickers = ['AMZN']

	###
	### PULL STOCK DATA
	###

	print("Pull stock data:\n")

	data = yf.download(  # or pdr.get_data_yahoo(...
	        # tickers list or string as well
	        tickers = tickers,

	        # use "period" instead of start/end
	        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
	        # (optional, default is '1mo')
	        period = period,

	        # fetch data by interval (including intraday if period < 60 days)
	        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
	        # (optional, default is '1d')
	        interval = "1d",

	        # group by ticker (to access via data['SPY'])
	        # (optional, default is 'column')
	        group_by = 'ticker',

	        # adjust all OHLC automatically
	        # (optional, default is False)
	        auto_adjust = False,

	        # download pre/post regular market hours data
	        # (optional, default is False)
	        prepost = False,

	        # use threads for mass downloading? (True/False/Integer)
	        # (optional, default is True)
	        threads = False,

	        # proxy URL scheme use use when downloading?
	        # (optional, default is None)
	        proxy = None
	    )
	return data

"""**Sharpe Ratio:**"""
def sharpeRatio(tickers, data, saveLocation, exchange, period):
	stockPrices = []
	stockReturnPctSeries = []
	stockStdDev = []
	stockAvgReturn = []
	stockSharpe = []
	stockReturnOverPeriod = [] # percent growth over the interval pulled. see "interval" in data yf.download object
	stockAvgVolume = []

	# Define function to pull first and last NaN from a list
	def firstNonNan(listfloats):
	  for item in listfloats:
	    if math.isnan(item) == False:
	      return item

	def lastNonNan(listfloats):
	  for i in range(len(listfloats)-1, 0, -1):
	    if math.isnan(listfloats[i]) == False:
	      return listfloats[i]

	# Convert the list of stocks to dictionary, to help with renaming dataframe rows
	# (dataframe rows must be renamed using dict and NOT list)
	stockListAsDict = {}
	for i in range(0,len(tickers)):
	  stockListAsDict[i] = tickers[i]

	#For each stock: calculate its sharpe ratio, store it in the list
	#Ex: if tickers is [MSFT, AAPL, SPY], then stockSharpe will be:
	#[MSFT sharpe ratio, AAPL sharpe ratio, SPY sharpe ratio]

	for i in progressbar(range(0,len(tickers)), "Computing Sharpe Ratio: ", 40): #progressbar for loop wrapper:
	  stockTicker = tickers[i]  
	  
	  stockPrices.append(round(data[stockTicker]['Close'],2))
	  stockReturnPctSeries.append(stockPrices[i].pct_change())
	  stockStdDev.append(stockReturnPctSeries[i].std())
	  stockAvgReturn.append(stockReturnPctSeries[i].mean())
	  
	  #Calculate return over period using helper firstNonNan and lastNonNaN functions
	  firstValidPrice = firstNonNan(stockPrices[i])
	  lastValidPrice = lastNonNan(stockPrices[i])
	  
	  try:
	    stockReturnOverPeriod.append((lastValidPrice - firstValidPrice) / firstValidPrice)
	  except:
	    print("An error occurred - Stock return over " + period)
	    stockReturnOverPeriod.append(float('NaN'))

	  stockAvgVolume.append(data[stockTicker]['Volume'].mean())
	  
	  #try:
	  stockSharpe.append((stockAvgReturn[i] - rfRate) / stockStdDev[i])
	  #except:
	  #  print("An error occurred - Sharpe ratio")
	  #  stockSharpe.append(float('NaN'))

	
	# Initialize dataframe, rename rows with stock symbols
	stockSharpeDailyDf = pd.DataFrame(stockSharpe,columns=["Daily Sharpe - " + period])
	stockSharpeDailyDf.rename(index=stockListAsDict, inplace=True)
	stockSharpeAnnualDf = stockSharpeDailyDf
	stockSharpeAnnualDf = stockSharpeAnnualDf.select_dtypes(exclude=['object', 'datetime']) * math.sqrt(252)
	stockSharpeAnnualDf.columns = ['Annual Sharpe - ' + period]
	#periodColumnName = "Return over " + period
	stockReturnOverPeriodDf = pd.DataFrame(stockReturnOverPeriod,columns=["Return over " + period + " (%)"])
	stockReturnOverPeriodDf.rename(index=stockListAsDict, inplace=True)
	stockAvgVolumeDf = pd.DataFrame(stockAvgVolume,columns=["Avg volume (" + period + ")"])
	stockAvgVolumeDf.rename(index=stockListAsDict, inplace=True)

	pd.concat([stockSharpeDailyDf, stockSharpeAnnualDf, stockReturnOverPeriodDf, stockAvgVolumeDf], 
	          axis=1).to_csv(saveLocation + '/stock_sharpe_' + exchange + '_' + datetime.now().strftime('%Y%m%d') + '_' + period + '.csv')

	print('Sharpe Ratio complete')
	return stockReturnPctSeries

# Covariance
def covariance(tickers, stockReturnPctSeries, saveLocation, exchange, period):
	### Move all stock returns into a dictionary, so that it can be added to a dataframe
	### (adding list to dataframe is hard; adding dict to DF is easy)

	stockDailyReturnPctDict = {}

	for i in range(0,len(tickers)):
	  stockDailyReturnPctDict[tickers[i]] = stockReturnPctSeries[i]

	stockReturnPctSeriesDf = pd.DataFrame(stockDailyReturnPctDict)
	stockReturnPctSeriesDf.to_csv(saveLocation + '/stockDailyReturnPctSeriesDf_' + exchange + '_' + datetime.now().strftime('%Y%m%d') + '_' + period + '.csv')

	###
	### Calculate stock covariance
	###
	stockCov = stockReturnPctSeriesDf.cov()
	#multiply by 100,000, as we are dealing with 2 percentages (* 100 * 100)
	stockCov = stockCov.select_dtypes(exclude=['object', 'datetime']) * 10000 
	stockCov.to_csv(saveLocation + "/stock_covariance_" + exchange + '_' + datetime.now().strftime('%Y%m%d') + '_' + period + ".csv")
	print("Covariance complete")


# Correlation
def correlation(tickers, stockReturnPctSeries, saveLocation, exchange, period):
	### Move all stock returns into a dictionary, so that it can be added to a dataframe
	### (adding list to dataframe is hard; adding dict to DF is easy)

	stockDailyReturnPctDict = {}

	for i in range(0,len(tickers)):
	  stockDailyReturnPctDict[tickers[i]] = stockReturnPctSeries[i]

	stockReturnPctSeriesDf = pd.DataFrame(stockDailyReturnPctDict)
	stockReturnPctSeriesDf.to_csv(saveLocation + '/stockDailyReturnPctSeriesDf_' + exchange + '_' + datetime.now().strftime('%Y%m%d') + '_' + period + '.csv')

	###
	### Calculate stock correlations
	###
	stockCorr = stockReturnPctSeriesDf.corr()
	stockCorr.to_csv(saveLocation + '/stock_correlation_' + exchange + '_' + datetime.now().strftime('%Y%m%d') + '_' + period + '.csv')
	print("Correlation complete")



def oldCode():
	print("oldCode() called")
	###
	###
	### OLD:
	### 
	### 
	#msftPrices = data.loc['2020-04-27':'2020-04-29'][("MSFT", "Open")]
	#msftPctSeries = msftPrices.pct_change()
	#print(msftPctSeries)

	#msftStdDev = msftPctSeries.std()
	#print("Std Dev: " + str(msftStdDev))

	#msftReturns = msftPctSeries.mean()
	#print("Avg daily return: " + str(msftReturns))

	#msftSharpe = (msftReturns - rfRate) / msftStdDev
	#print("Sharpe ratio: "+str(msftSharpe))

	###
	###
	### REFERENCE:
	### 
	### 
	# msft = yf.Ticker("MSFT")
	# get stock info
	# msft.history(period="max")

	# msftPrices.mean() 
	# msftPrices.std()
	# Sharpe_Ratio = portf_val[『Daily Return'].mean() / portf_val[『Daily Return'].std()

	# type(data)
	# data.info()
	# data.loc['2020-04-27'][("MSFT", "Open")]
	# data.loc['2020-04-27'][("MSFT", slice(None))]
	return





def main():
	import argparse

	parser = argparse.ArgumentParser()
	#parser.add_argument("FILE", help="File to store as Gist", nargs="+")

	# Arguments:
	# -s: stockExchange .csv file (looks up files in /home/pi/dev/data/reference) ex: -s asx (no need to include ".csv")
	parser.add_argument('-s', action='append', dest='stockExchange',
	                    default=[],
	                    help='Add repeated stock exchange lists to a list (ex: NYSE, NASDAQ)',
	                    )

	# -p: period... ex: -p 1mo
	parser.add_argument('-p', action='append', dest='periodVar',
	                    default=[],
	                    help='Add periods to a list (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)'
	                    )

	# -e: exclude correlation or covariance from being generated.
	# three possible arguments: -e correlation, -e covariance. No "-e sharperatio", sharpe ratio must always be generated.
	parser.add_argument('-e', action='append', dest='excludedFiles',
	                    default=[],
	                    help='Add repeated stock exchange lists to a list (ex: NYSE, NASDAQ)',
	                    )

	args = parser.parse_args()
	stockExchange = args.stockExchange
	periodVar = args.periodVar
	excludedFiles = args.excludedFiles # specify files that should NOT be generated, ex: do not generate correlation

	# Verify excludedFiles parameters are correct
	for i in range(0,len(excludedFiles)):
		if excludedFiles[i].lower() == 'correlation':
			excludedFiles[i] = 'correlation' # rewrite with lower case version
			continue
		elif excludedFiles[i].lower() == 'covariance':
			excludedFiles[i]= 'covariance'
			continue
		else:
			raise Exception("Incorrect -e entered. Accepted values: correlation, covariance [case insensitive]")

	global rfRateAnnualized
	rfRateAnnualized = 0.0012 # 3 month treasury bill rate, https://ycharts.com/indicators/3_month_t_bill
	global rfRate
	rfRate = (rfRateAnnualized/252) 
	#periodVar = '1mo'
	saveLocation = '/home/pi/dev/data/'
	#tickers = ['GOOG','AAPL','BBRY']
	#stockExchange = 'nasdaq'

	for exchange in stockExchange:
		for period in periodVar:
			tickers = readStockTickers(exchange) # change this as needed
			print(tickers)

			data = downloadYFinanceData(tickers, rfRateAnnualized, rfRate, period)
			stockReturnPctSeries = sharpeRatio(tickers, data, saveLocation, exchange, period)
			if 'correlation' not in excludedFiles:
				correlation(tickers, stockReturnPctSeries, saveLocation, exchange, period)

			if 'covariance' not in excludedFiles:
				covariance(tickers, stockReturnPctSeries, saveLocation, exchange, period)

			### PRINT PARAMETERS:
			startDate = data.index[1]
			endDate = data.index[-1]

			print('\n')
			print("Number of stocks: " + str(len(tickers)))
			print("Risk free rate (annualized): " + str(rfRateAnnualized*100) + '%')
			print('\n')
			print("Start date: " + startDate.strftime('%Y-%m-%d'))
			print("End date: " + endDate.strftime('%Y-%m-%d'))

			# Upload
			todayDate = datetime.now().strftime('%Y%m%d')
			scp.scpToServer('/home/pi/dev/data/stock_sharpe_' + exchange + '_' + todayDate + '_' + period + '.csv', upload_ftp_location)

			if 'covariance' not in excludedFiles:
				scp.scpToServer('/home/pi/dev/data/stock_covariance_' + exchange + '_' + todayDate + '_' + period + '.csv', upload_ftp_location)

			if 'correlation' not in excludedFiles:
				scp.scpToServer('/home/pi/dev/data/stock_correlation_' + exchange + '_' + todayDate + '_' + period + '.csv', upload_ftp_location)

main()
