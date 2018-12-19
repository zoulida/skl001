__author__ = 'zoulida'
from concurrent.futures import ThreadPoolExecutor,wait,as_completed
import urllib.request
URLS = ['http://www.163.com', 'https://www.baidu.com/', 'https://github.com/']
def load_url(url):
    with urllib.request.urlopen(url, timeout=60) as conn:
        print('%r page is %d bytes' % (url, len(conn.read())))
        list_a.append('ddd')
    return len(conn.read())

list_a = []
executor = ThreadPoolExecutor(max_workers=3)

f_list = []
for url in URLS:
    future = executor.submit(load_url,url)
    f_list.append(future)
print(wait(f_list))

print('主线程')
for l in list_a:
    print(l)