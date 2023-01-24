# pip install deepface
from deepface import DeepFace
#импортируем json что бы легче было читать
import json

def face_verefy(img_1, img_2):          #прроверяем соответствует ли лицо одному человеку
    try:
        result_dict = DeepFace.verify(img1_path=img_1, img2_path=img_2, model_name='Facenet512', enforce_detection=False)
        with open('result.json', 'w') as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)  # записываем в файл
        # return result_dict
        if result_dict.get('verified'):
            return 'Проверка пройдена. Пропустить.'
        return 'Нарушитель! Задержать!!!!'
    except Exception as _ex:
        return _ex

#Пишем функцию рассаознавания лица
def face_recogn():
    try:
        result = DeepFace.find(img_path='face/if_1.jpg', db_path='face', enforce_detection=False)
        result = result.values.tolist()
        return result
    except Exception as _ex:
        return _ex
#распознавание пол возраст эмоции

def face_analyze():
    try:
        result_dict = DeepFace.analyze(img_path='name_photo', actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)
        with open('face_analyze.json', 'w') as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)  # записываем в файл
        print(f'[+] Age: {result_dict.get("age")}')
        print(f'[+] Gander: {result_dict.get("gender")}')
        print(f'[+] Emotion: {result_dict.get("emotion")}')
        print(f'[+] Race: {result_dict.get("race")}')
        for k, v in result_dict.get('emotion').items():
            print(f'{k} - {round(v, 2)}%')
        # return result_dict

    except Exception as _ex:
        return _ex


def main():
    # print(face_verefy(img_1='face/if_1.jpg', img_2='face/if_3.jpg'))
    # print(face_recogn())
    print(face_analyze())
if __name__ == '__main__':
    main()