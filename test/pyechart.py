__author__ = 'zoulida'
#from future.types import disallow_types
from _pytest.dbQueryPools import queryMySQL_plot_stock_market as pm
from zldtest.dbQueryPools import queryMySQL_plot_stock_market as pm
from future17.dbQueryPools import queryMySQL_plot_stock_market as pm
from future17.utils import viewitems
from pyecharts import Bar
bar = Bar("我的第一个图表", "这里是副标题")
bar.add("服装", ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"], [5, 20, 36, 10, 75, 90])
bar.show_config()
bar.render()