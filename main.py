from PIL import Image
from yolo import YOLO

def main():
    yolo = YOLO()

    while True:
        image_path = input('Input image filename:')
        try:
            image = Image.open(image_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            i = yolo.detect_image(image.crop((290, 400, 1330, 900)))
            i.save('a.jpg')

    yolo.close_session()

main()
