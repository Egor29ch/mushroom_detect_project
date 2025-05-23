import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

# Определение и анализ пользовательских аргументов, вводимых пользователем
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Путь к файлу модели YOLO (например: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Источником изображения может быть файл изображения ("test.jpg"), \
                    папка с изображениями ("test_dir"), видеофайл ("testvid.mp4") или индекс USB-камеры ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Минимальный доверительный порог для отображения обнаруженных объектов (например: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Разрешение в WхH для отображения результатов логического вывода при (например: "640x480"), \
                    в противном случае используйте разрешение, соответствующее исходному',
                    default=None)
parser.add_argument('--record', help='Запишите результаты с видео или веб-камеры и сохраните их как "demo1.avi". Для записи необходимо указать параметр --resolution.',
                    action='store_true')

args = parser.parse_args()


# Анализировать вводимые пользователем данные
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Проверить, существует ли файл модели и является ли он допустимым
if (not os.path.exists(model_path)):
    print('ERROR: Указан неверный путь к модели или модель не была найдена. Убедитесь, что имя файла модели введено правильно.')
    sys.exit(0)

# Загрузить модель в память и получите названия классов
model = YOLO(model_path, task='detect')
labels = model.names

# Проанализировать входные данные, чтобы определить, является ли источником изображения файл, папка, видео или USB-камера
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Расширение файла {ext} не поддерживается.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Ввод {img_source} неверен. Пожалуйста, попробуйте снова.')
    sys.exit(0)

# Анализ разрешения экрана, заданного пользователем
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Проверить правильность записи и настройте запись
if record:
    if source_type not in ['video','usb']:
        print('Запись работает только с видеоисточниками и камерами. Пожалуйста, попробуйте снова.')
        sys.exit(0)
    if not user_res:
        print('Пожалуйста, укажите разрешение для записи видео.')
        sys.exit(0)
    
    # Настройка записи
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Загрузка или инициализация источника изображения
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Установление разрешение камеры или видео, если оно указано пользователем
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Установление цвета ограничивающей рамки (используя цветовую схему Tableau 10)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Инициализирование управляющей переменной 
img_count = 0

# Создание флажков и инициализирование переменных для вывода информационной справки
stop_if_Entoloma_Lividum = False
object_count_Entoloma_Lividum = 0

stop_if_Amanita_Pantherina = False
object_count_Amanita_Pantherina = 0

stop_if_Hydnum_Rufescens = False
object_count_Hydnum_Rufescens = 0

# Начало цикла логического вывода
while True:

    t_start = time.perf_counter()

    # Загрузка кадра из источника изображения
    if source_type == 'image' or source_type == 'folder': # Если источником является изображение или папка с изображениями, загрузить изображение, используя его имя файла
        if img_count >= len(imgs_list):
            print('Все изображения были обработаны. Выход из программы.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': # Если источником является видео, загрузить следующий кадр из видеофайла
        ret, frame = cap.read()
        if not ret:
            print('Конец видеофайла. Выход из программы.')
            break
    
    elif source_type == 'usb': # Если источником является USB-камера, захватить кадр с камеры
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Не удается считывать кадры с камеры. Это означает, что камера отключена или не работает. Выход из программы.')
            break

    elif source_type == 'picamera': # Если источником является Picamera, захватить кадры с помощью интерфейса picamera
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Не удается считывать кадры с камеры Picamera. Это означает, что камера отключена или не работает. Выход из программы.')
            break

    # Изменить размер рамки до желаемого разрешения экрана
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Выполнить вывод по кадру
    results = model(frame, verbose=False)

    # Извлекать результаты
    detections = results[0].boxes

    # Пройти через каждое обнаружение и получить координаты ограничивающего прямоугольника, уверенность и класс
    for i in range(len(detections)):

        # Получить координаты ограничивающего прямоугольника
        # Ultralytics возвращает результаты в тензорном формате, которые должны быть преобразованы в обычный массив Python  
        xyxy_tensor = detections[i].xyxy.cpu() # Обнаружения в тензорном формате в памяти процессора
        xyxy = xyxy_tensor.numpy().squeeze() # Преобразование тензоров в числовой массив
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Извлечь отдельные координаты и преобразовать их в int

        # Получить идентификатор и имя класса ограничивающей рамки
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        
        # Получить уровень доверия
        conf = detections[i].conf.item()

        # Нарисовать рамку, если доверительный порог достаточно высок
        if conf > 0.8:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Получить размер шрифта
            label_ymin = max(ymin, labelSize[1] + 10) # Следить за тем, чтобы надпись не располагалась слишком близко к верхней части окна
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Нарисовать белый прямоугольник, в который нужно поместить текст надписи
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Нарисовать текст надписи

            # Счётчик объектов определенного класса
            if classname == "Entoloma_Lividum":
                object_count_Entoloma_Lividum += 1
            elif classname == "Amanita_Pantherina":
                object_count_Amanita_Pantherina += 1
            elif classname == "Hydnum_Rufescens":
                object_count_Hydnum_Rufescens += 1
            
            
            if object_count_Entoloma_Lividum >=1 and not stop_if_Entoloma_Lividum:
                root = tk.Tk()
                root.title("Информационное сообщение")
                
                # Установка размера окна
                root.geometry("500x400")
                
                # Загрузка изображения (замените путь на свой)
                try:
                    img = Image.open("entoloma_lividum_pic.jpg")  # Укажите путь к вашему изображению
                    img = img.resize((200, 200), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    # Создание виджета для изображения
                    image_label = ttk.Label(root, image=photo)
                    image_label.image = photo  # сохраняем ссылку на изображение
                    image_label.pack(pady=10)
                except FileNotFoundError:
                    error_label = ttk.Label(root, text="Изображение не найдено", foreground="red")
                    error_label.pack(pady=10)
                
                # Заголовок сообщения
                title_label = ttk.Label(root, text="Энтолома ядовитая", font=("Arial", 14, "bold"))
                title_label.pack(pady=5)
                
                # Текстовое описание
                description_text = """Энтолома ядовитая (или розовопластинник ядовитый) — ядовитый вид грибов рода энтолома.

При употреблении этот гриб раздражает слизистую оболочку желудочно-кишечного тракта, вызывая «резиноидный синдром» (боли в животе, рвота, жидкий стул).(несъедобный)"""
                
                description_label = ttk.Label(root, text=description_text, wraplength=400, justify="center")
                description_label.pack(pady=10, padx=20)
                
                # Кнопка закрытия
                close_button = ttk.Button(root, text="Закрыть", command=root.destroy)
                close_button.pack(pady=10)
                
                # Запуск главного цикла
                root.mainloop()
                stop_if_Entoloma_Lividum = True
            elif object_count_Amanita_Pantherina >=1 and not stop_if_Amanita_Pantherina:
                root = tk.Tk()
                root.title("Информационное сообщение")
                
                # Установка размера окна
                root.geometry("500x400")
                
                # Загрузка изображения (замените путь на свой)
                try:
                    img = Image.open("amanita_pantherina_pic.jpg")  # Укажите путь к вашему изображению
                    img = img.resize((200, 200), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    # Создание виджета для изображения
                    image_label = ttk.Label(root, image=photo)
                    image_label.image = photo  # сохраняем ссылку на изображение
                    image_label.pack(pady=10)
                except FileNotFoundError:
                    error_label = ttk.Label(root, text="Изображение не найдено", foreground="red")
                    error_label.pack(pady=10)
                
                # Заголовок сообщения
                title_label = ttk.Label(root, text="Мухомор пантерный", font=("Arial", 14, "bold"))
                title_label.pack(pady=5)
                
                # Текстовое описание
                description_text = """Мухомор пантерный (Amanita pantherina) — гриб рода мухомор семейства аманитовые.

Шляпка диаметром 4–12 см, плотная, сначала полусферическая, затем выпуклая и полностью распростёртая, с тонким рубчатым краем и иногда небольшими свисающими хлопьями. Кожица буроватого цвета, гладкая и блестящая, покрыта мелкими белыми хлопьями.(несъедобный)"""
                
                description_label = ttk.Label(root, text=description_text, wraplength=400, justify="center")
                description_label.pack(pady=10, padx=20)
                
                # Кнопка закрытия
                close_button = ttk.Button(root, text="Закрыть", command=root.destroy)
                close_button.pack(pady=10)
                
                # Запуск главного цикла
                root.mainloop()
                stop_if_Amanita_Pantherina = True
            elif object_count_Hydnum_Rufescens >=1 and not stop_if_Hydnum_Rufescens:
                root = tk.Tk()
                root.title("Информационное сообщение")
                
                # Установка размера окна    
                root.geometry("500x400")
                
                # Загрузка изображения (замените путь на свой)
                try:
                    img = Image.open("Hydnum_rufescens_ pic.JPG")  # Укажите путь к вашему изображению
                    img = img.resize((200, 200), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    # Создание виджета для изображения
                    image_label = ttk.Label(root, image=photo)
                    image_label.image = photo  # сохраняем ссылку на изображение
                    image_label.pack(pady=10)
                except FileNotFoundError:
                    error_label = ttk.Label(root, text="Изображение не найдено", foreground="red")
                    error_label.pack(pady=10)
                
                # Заголовок сообщения
                title_label = ttk.Label(root, text="Ежовик рыжеющий", font=("Arial", 14, "bold"))
                title_label.pack(pady=5)
                
                # Текстовое описание
                description_text = """Плодовые тела одиночные или в сростках, шляпко-ножечные. Шляпка 2—5(10) см в поперечнике, выпуклая, затем уплощённая и с широким неглубоким понижением в центральной части, у молодых грибов с подвёрнутым краем, в молодом возрасте несколько бархатистая. Окраска охристая, оранжево-жёлтая или оранжево-коричневая, с возрастом плодового тела бледнеет, иногда имеются концентрические зоны.(Съедобный)"""
                
                description_label = ttk.Label(root, text=description_text, wraplength=400, justify="center")
                description_label.pack(pady=10, padx=20)
                
                # Кнопка закрытия
                close_button = ttk.Button(root, text="Закрыть", command=root.destroy)
                close_button.pack(pady=10)
                
                # Запуск главного цикла
                root.mainloop()
                stop_if_Hydnum_Rufescens = True
    
    # Отображение результатов обнаружения
    cv2.imshow('Результаты обнаружения YOLO',frame) # Отображаемое изображение
    if record: recorder.write(frame)

    # Если делать вывод по отдельным изображениям, нужно дождаться нажатия клавиши пользователем, прежде чем переходить к следующему изображению. В противном случае нужно подождать 5 мс, прежде чем переходить к следующему кадру.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Нажать, чтобы выйти
        print("Вы вышли из программы")
        break
    elif key == ord('s') or key == ord('S'): #  Нажать "s", чтобы поставить на паузу
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): #  Нажать "p", чтобы сохранить изображение результатов на этом кадре
        cv2.imwrite('capture.png',frame)
    
# Конец кода
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
