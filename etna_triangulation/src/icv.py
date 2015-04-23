from icv import *



if __name__ == "__main__":

    img1 = read_image('data/pattern1.png')
    img1 = bgr2gray(img1)
    img1 = scale_image(img1, scale=0.5)
    show_image(img1, cmap='gray')

    img = read_image('data/opencv.png')
    img = bgr2gray(img)
    show_image(skeleton(img))

    webcam = Webcam(device=1)
    webcam.run()
#    webcam.set_size((1280, 720))
#    webcam.set_parameters(0.30, 0.15, 0.10)
#    webcam.run(callback=lambda img: calibration.draw_location_results(img, webcam.frame_rate()))

    #camera = Camera()
    #camera.run()