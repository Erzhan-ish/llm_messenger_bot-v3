import requests


url = "https://b24-kyul8c.bitrix24.ru/rest/53/0f52nngl44abqy7t/crm.deal.list.json"



response = requests.get(url)
print(response.json())
