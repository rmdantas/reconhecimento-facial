'''
PIP'S!!
(WINDOWS)
pip install opencv-python
pip install opencv-contrib-python
pip install numpy

(LINUX/MAC)
pip3 install opencv-python
pip3 install opencv-contrib-python
pip3 install numpy
'''

'''
    Bibliotecas importadas:
    cv2 -> Biblioteca para tratamento de imagens/vídeos.
    os -> Biblioteca para criação de pastas.
    numpy -> Biblioteca para trabalharmos com arrays das imagens/vídeos.
'''
import cv2
import os
import numpy as np

print("Aperte *s* para inserir o seu nome\nDepois aperte *t* para começar o algoritmo de treinamento de imagem\nE voilà\nAperte *q* para fechar o programa")

def savePerson():
    '''
        Função que salva o nome da pessoa ná variavel global.
    '''
    global ultimoNome
    global boolsaveimg
    name = input('Qual o seu nome: ')
    ultimoNome = name
    boolsaveimg = True

def saveImg(img):
    '''
        Função que rebera uma imagem e à salvara em uma pasta neste mesmo ditório
    '''
    global ultimoNome

    if not os.path.exists('train'):
        os.makedirs('train')
    
    if not os.path.exists(f'train/{ultimoNome}'):
        os.makedirs(f'train/{ultimoNome}')

    files = os.listdir(f'train/{ultimoNome}')

    cv2.imwrite(f'train/{ultimoNome}/{str(len(files))}.jpg', img)

def trainData():
    '''
        Função que ira treinar o nosso algoritmo de reconhecimento e dira de quem este rosto pertence.
        Ele entrara em todas as pastas criadas reconhecera na camera quem é ela, uma vez que a pessoa já foi cadastrada no 
        detector 
    '''
    global trained
    trained = True
    global recognizer
    global persons
    persons = os.listdir('train')

    ids = []
    faces = []

    for i, p in enumerate(persons):
        for f in os.listdir(f'train/{p}'):
            img = cv2.imread(f'train/{p}/{f}', 0)
            faces.append(img)
            ids.append(i)

    recognizer.train(faces, np.array(ids))

# Váriaveis globais
ultimoNome = ''
boolsaveimg = False
savecount = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
trained = False
persons = []

# Chamando minha webcam
#obs, use '1' ou '0' se a sua webcam for integrada
webcam = 'https://192.168.15.15:8080/video'
cap = cv2.VideoCapture(webcam)

# Carregando o **haar cascade** 
# Nosso arquivo xml responsavel pelo treinamento de detecção de face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    '''
        Aqui é onde a mágia aconteçe. 
        Aqui é onde abriremos a webcam, tiraremos as fotos dos rostos cortados para detecção, passaremos por elas para o algoritmo entender quem é quem.
    '''
    ret, frame = cap.read()

    # Modelando a imagem para ela interpretar o vídeo como cinza, para melhor detecção
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detecta método que detecta rostos em nosso algoritmo
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    # Aqui corremos em todas os rostos encontrados
    for (x, y, w, h) in faces:
        
        # Criando um ROI/Cortando a imagem para melhor refinamento
        roi = gray[y:y+h, x:x+w]
        # Logo após o roi, daremos um resize nele para uma maior refinação
        roi = cv2.resize(roi, (50, 50))

        # Desenhando um Retângulo em volta do rosto
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        if trained:
            # Se o rosto aqui foi achado sera posto o nome da pessoa na tela
            idf, conf = recognizer.predict(roi)
            nameP = persons[idf]
            cv2.putText(frame, nameP, (x, y), 2, 1,(0,255, 0), cv2.FONT_HERSHEY_SIMPLEX, True)

        if boolsaveimg:
            # Salvando a imagem cortada no ditório criado
            saveImg(roi)
            savecount  += 1

        if savecount > 50:
            # Determinando uma quantidade exata de imagens que pode ser salvada
            boolsaveimg = False
            savecount = 0

    # Mostrando a imagem cinza
    cv2.imshow('Cinza', gray)


    if ret:
        # Se houver uma webcam, conecte
        cv2.imshow('WEBCAM', frame)
        
    # Aquardando uma chave.
    key = cv2.waitKey(1)

    if key == ord('s'):
        # Chamando a função que salvara a imagem
        savePerson()

    if key == ord('t'):
        # Chamando a função que ira treinar em cima das imagens
        # Ela é a responsável por descobrir as pessoas identificadas
        trainData()

    if key == ord('q'):
        break

# Liberando o cache e fechando todas as janelas
cap.release()
cv2.destroyAllWindows()