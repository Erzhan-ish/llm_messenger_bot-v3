import requests

url = "https://b24-kyul8c.bitrix24.ru/rest/53/0f52nngl44abqy7t/crm.activity.type.list.json"
print(requests.get(url).json())