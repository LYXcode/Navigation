from bs4 import BeautifulSoup
from urllib.request import urlopen

import re
# html = urlopen("https://morvanzhou.github.io/static/scraping/list.html").read().decode('utf-8')
# soup = BeautifulSoup(html, features='lxml')
# month = soup.find_all("ul", {"class": "jan"})
# for i in month:
#     print(i.get_text())

string = "dog aun\ns to cat ran r2n"
print(string)
pattern1 = "dog"
pattern2 = "bird"
print(re.search(r"[0-9a-z]", string))
print(re.search(r"Monday?", "Monda Monday"))