import tushare as ts
df = ts.get_h_data('600848') #前复权
# df = ts.get_h_data('600848',autype='hfq') #后复权，可以设定开始和结束日期
df = ts.realtime_boxoffice()
dfn = ts.get_stock_basics()
print(dfn)

