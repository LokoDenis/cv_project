import requests
from html.parser import HTMLParser

class kinopoiskParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        desired = False
        if (tag == 'div'):
            for attr in attrs:
                if (attr[0] == 'class' and (attr[1] == 'el' or attr[1] == 'el even')):
                    desired = True
        if desired:
            for attr in attrs:
                if (attr[0] == 'id'):
                    image_name = attr[1][3:]
                    req = requests.get("http://st.kp.yandex.net/images/film_big/" + image_name + ".jpg")
                    image = open(image_name + ".jpg", "wb")
                    print("http://st.kp.yandex.net/images/film_big/" + image_name + ".jpg")
                    image.write(req.content)
                    image.close()



for i in range(1, 51):
    r = requests.get('http://www.kinopoisk.ru/popular/day/2016-05-30/page/' + str(i) + '/', params = None,
                    headers={
            'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686; ru; rv:1.9.1.8) Gecko/20100214 Linux Mint/8 (Helena) Firefox/3.5.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru,en-us;q=0.7,en;q=0.3',
            'Accept-Encoding': 'deflate',
            'Accept-Charset': 'windows-1251,utf-8;q=0.7,*;q=0.7',
            'Keep-Alive': '300',
            'Connection': 'keep-alive',
            'Referer': 'http://www.kinopoisk.ru/',
            'Cookie': 'users_info[check_sh_bool]=none; search_last_date=2010-02-19; search_last_month=2010-02;                                        PHPSESSID=b6df76a958983da150476d9cfa0aab18',
        })

    print(r)
    kp = kinopoiskParser()
    kp.feed(r.text)
