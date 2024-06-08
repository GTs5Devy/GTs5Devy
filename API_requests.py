import requests
def fetch(user,pas):
    url = 'https://homeaccesscenterapi.vercel.app/api/'

    data = {
        'username': user,
        'password': pas
    }

    try : 
        #response = requests.post(url,json=data)
        response = requests.get(url,json=data)
        response.raise_for_status()

        print(response.json())
        
    except requests.exceptions.RequestException as e:
        print(e)

fetch("s531055","goat9616")