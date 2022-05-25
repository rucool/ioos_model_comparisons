import re

text =  '2020/12/10 11:50:02 ELTi_LOVE_2020_12_10_1500.ruv 69502'
pattern = '(\d+/\d+/\d+ \d+:\d+:\d+) ((RDL|ELT)._(.*)_\d{4}_\d{2}_\d{2}_\d{4}.ruv) (\d{5,9})'

ans = re.search(pattern, text) #save to groups