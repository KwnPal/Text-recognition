import json


f = open('data.json')

data = json.load(f)

for i in data["cats"]:
    print(i)

f.close()