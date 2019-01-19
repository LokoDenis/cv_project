# README #

### Ad Search Project ###

* This is the project based on The OPENCV library. Supporting by Denis Belyakov, 1st course student of the faculty of Computer Science, Higher School of Economics.

* The aim of this project is to create a program (or maybe an application), which will be able to show images from a base looking the same to the inputted one. It involves doing research in the Computer Vision sphere and learning about basic operations with pictures. 

Guided by Vadim Gorbachev.




### About source files and directories ###

* first_steps.cpp and "first_steps" dir - includes examples of basic operation with images, such as rotation and color inversion. 
* detectors.cpp and "detectors" dir - includes examples of detectors' and descriptors' work plus matching.
* clustering.cpp and "clusters" dir - includes examples of clustering on one image.
* grads.cpp and "gradients" dir - shows the look of gradient vectors on pictures
* "report" folder contains report on my results.
* "helpful_vision" folder contains helpful project files, such as building a base, resizing images, cleaning the source from corrupted pictures and etc.
* "bot" directory contains files related to parsing web-pages, downloading pictures from yandex servers and coordinating the bot's job. 

**The main file**
search.cpp - core of the bot.

* "data" directory includes examples of YAML files being saved.

* capable of searching for images through the base

** For more detailed information related to the current progress consider looking through issues and changelog.txt **

### Visualization ###


![Search.png](https://bitbucket.org/repo/6GEkA7/images/369375486-Search.png)


* All the work done can be seen using the Telegram Bot @MoviePosterBot

* It is capable of receiving the photo of a movie ad from chat and giving a link to Kinopoisk as a result.


### Оптимизации и некоторые данные касательно работы программы. ###

**Вся информация по прогрессу в разработке программы содержится в виде отчета в папке report и в виде issues**

1) Количество точек.

Для нахождения особых точек в конечном счете был выбран детектор SIFT с параметрами, позволяющими находить в среднем на хорошем иображении около 700-900 особых точек. Выбор таких чувствительных параметров детектора обусловлен нахождением в базе очень сложных для выявления особых точек изображений (пример снизу). Возникла проблема, как одним детектором найти адекватное количество точек и на плохих, и на хороших картинках.
Решением является выделение лучших точек на основе их отклика и размера выделяемой области (за это отвечают поля response и size структуры KeyPoint). В итоге, вектор точек сортируется по данным полям, а если их количество слишком большое, выбираются 300 лучших. Это позволяет и повысить точность поиска, и уменьшить вес.

![one.jpg](https://bitbucket.org/repo/6GEkA7/images/3948984803-one.jpg)

2) Использующееся количество кластеров — 1100.  Эксперементально полученное оптимальное значение исходя из занимаемого базой места, скорости подсчета базы (около часа) и точности работы.

3) Предварительная обработка изображений.
Размер каждой картинки в базе был уменьшен до 480*640, что позволяет детектору выделять более крупные и значимые области. 
Однако ключевой деталью, позволяющей программе работать весьма точно, оказалось размытие изображений. Для каждого изображения из базы я использую GaussianBlur. Его значимость обсуславливается тем, что во-первых, фотография редко бывает особо точной и резкой, а во-вторых, наличием на некоторых изображений точек, которые не несут в себе много информации касательно всего изображения, но имеют очень сильный response. Например, черные точки на платьях на картинке ниже.

![sea.jpg](https://bitbucket.org/repo/6GEkA7/images/2619690812-sea.jpg)
   
Также размывается и изображение, подаваемое на вход, чтобы снизить значение точек, не относящиеся к запечатленным на них афишам (посторонние  предметы вроде клавиатуры и бликов экрана и т.п.)

3) Инвертированный индекс.

Главный оптимизацией базы, позволяющей снизить количество картинок в инвертированном индексе является увеличение порога для ненулевого элемента при подсчете по TF-IDF метрике. Если, например, взять, значение дельты, равное 0.02, то это серьезно повысит эффективность по количеству сравнений, но снижение точности работы программы в некоторых случаях оказывается критичным. 

4) Для переранжирования берутся лучшие 60 изображений, но можно брать и меньше, что несколько уменьшит время работы программы. В среднем, нужная картинка находится в этом топе.

5) Отношение количества инлаеров к количеству лучших заматченных точек для ответа обычно составляет 90 — 100 процентов.

6) Размер базы при использовании стандартных методов библиотеки OpenCV в моем случае можно снизить до 465 мегабайт. Это включает в себя по файлу дескриптора, визуального слова и вектора кейпоинтов на каждое изображение, файл для построения инвертированного индекса, файл центров кластеров и файл для вычисления TF-IDF метрики.

Для лучшего показателя по памяти используется сжатие .gz

![base.jpg](https://bitbucket.org/repo/6GEkA7/images/1644731146-base.jpg)

![base.jpg](https://bitbucket.org/repo/6GEkA7/images/1885595572-base.jpg)

7) Среднее время работы программы на локальном устройстве составляет 5-8 секунд. Среднее время на среднем дроплете DigitalOcean - 15-16 сек.


### Who do I talk to? ###

* The owner of this repo is sanityseeker
* Contact emails: 
sanityseeker@ya.ru, dobelyakov@edu.hse.ru
