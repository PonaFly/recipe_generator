import requests
import os
import json
import pickle
from bs4 import BeautifulSoup

def load_page(url,page,session,save_path): 
    url += str(page) 
    request = session.get(url)
    with open(save_path + '/page_%d.html' % (page), 'w') as output_file:
        output_file.write(request.text)
    return request

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        

s = requests.Session() 
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Linux; U; Android 2.2) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1'
    })

json_file = open('private_data.json', 'r')
food_url = json_file['food_url']
json_file.close()
vypecka_deserty_folder = './vypechka-deserty'
fields_needed = ['recipeIngredient', 'name', 'recipeInstructions']

i = 1
result = []
while True:
    print('downloading page #', i)
    page = load_page(food_url, i, s, vypecka_deserty_folder)
    if page.status_code == 404:
        break
        
    soup = BeautifulSoup(page.text)
    data = json.loads(soup.find('script', type='application/ld+json').text)['itemListElement']
    for recipe in data:
        temp_dict = {}
        for field in fields_needed:
            temp_dict[field] = recipe[field]
        result.append(temp_dict)
    i += 1
    
save_obj(result,'data_bot')

