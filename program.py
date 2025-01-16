from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pygame, sys
from imutils import resize

# if ou want it shows image in form of array (currently uncalled)
def show(image):

    plt.imshow(image, cmap=plt.get_cmap('gray'))  
    plt.show()          #shows image as a graph
    print(image)


# function to make a neural network
def make_model(data, epochs):
    from keras.layers import Flatten, Dense, Dropout
    from keras.models import Sequential

    (x_train, y_train), (x_test, y_test) = data.load_data()         # splitting data into x, y and training and testing
    x_train, x_test = x_train / 255, x_test / 255       # normalistation

    model = Sequential([
    Flatten(input_shape=(28, 28)),     # image into 28*28 = 784 nodes
    Dense(128, activation='relu'),     # hidden layer of 128 nodes
    Dropout(0.2),                      # dropout layer to prevent overfitting
    Dense(10, activation='softmax')    # output layer of 10 nodes for 10 classes
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    model.fit(x_train, y_train, epochs=epochs)       # training the model

    model.evaluate(x_test, y_test)      # evaluation

    model.save('model.h5')      # saving to .h5


# prediction function
def predict(input_image, ml_model):
    predicted_probabilities = ml_model(np.expand_dims(input_image, axis=0))     # gives probability of every class
    pred = np.argmax(predicted_probabilities, axis=1)[0]                        # finds most probable prediction
    percentage = int(predicted_probabilities[0, pred] * 10000) / 100            # probability of that prediction
    output = f"{pred} : {percentage}%"      # prediction
    return str(output)


# taking from camera
def from_cam(model):
    prediction = "" # prediction variable
    on = False      # checks if prediciton is on or not
    cam = cv2.VideoCapture(0)       # Uses primary video source

    print("Instructions:\n1) Use light background and dark pen and provide adequade light\n2) Show it to your webcam and place number in the box\n3) press x to exit")

    while True:
        _, img = cam.read() #img stores camera feed

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # grayscale image
        input_img = input_img[170:310, 250:390]         # cropping
        input_img = cv2.threshold(input_img, 125, 255, cv2.THRESH_BINARY_INV)[1]  #thresholding
        input_img = resize(input_img, 28, 28)     # final input image

        cv2.putText(img, prediction, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)    # displaying prediction

        cv2.rectangle(img, (250, 170), (390, 310), (0, 255, 0), 3)

        cv2.imshow("input", cv2.resize(input_img, (280, 280)))  # model input
        cv2.imshow("cam", img)  # actual camera feed
    
        key = cv2.waitKey(1)  # checks key if pressed

        if key == ord('x'): # exit
            break 
            
        # prediction mechanism
        try:
            prediction = predict(input_img, model)
        except:
            print("failed")
            break

    cam.release()
    cv2.destroyAllWindows()     # removing screen

    main(model)      # returing to main menu


# taking from a drawable screen
def canvas(model):
    print('Instructions: \n1) Draw using mouse\n2) press r to redraw \n3) click the close button to close.')
    pygame.font.init()

    dim = 28    # shape of drawn input
    screen = pygame.display.set_mode((dim*15, dim*15))
    pygame.display.set_caption('canvas')
    
    
    draw_on = False
    last_pos = (0, 0)
    arr = np.zeros((dim*15, dim*15))  # array for drawn image

    radius = 10
    white = (255, 255, 255)
    font = pygame.font.Font(None, 36)

    # rendering text on screen
    def render_text(message, x, y):
        text_surface = font.render(message, True, (255, 255, 255))
        screen.blit(text_surface, (x, y))

    # this functon draws circles in a line, acting as a bruch tool
    def roundline(canvas, color, start, end, radius=1):
        Xaxis = end[0]-start[0]
        Yaxis = end[1]-start[1]
        dist = max(abs(Xaxis), abs(Yaxis))  # the distance between the points
        for i in range(dist):
            x = int(start[0]+float(i)/dist*Xaxis)
            y = int(start[1]+float(i)/dist*Yaxis)
            pygame.draw.circle(canvas, color, (x, y), radius)   # circle drawing

    run = True
    while run:
        event = pygame.event.wait()     # checks the events that occur

        if event.type == pygame.QUIT:   # to quit
            run = False
    
        if event.type == pygame.MOUSEBUTTONDOWN:        # user is drawing
            draw_on = True
        
        if event.type == pygame.MOUSEBUTTONUP:          # user is not drawing
            draw_on = False

        if pygame.key.get_pressed()[pygame.K_r]:        # redrawing mechanism
            canvas(model)
        
        # drawing mechanism
        if event.type == pygame.MOUSEMOTION:            
            if draw_on:
                pygame.draw.circle(screen, white, event.pos, radius)
                roundline(screen, white, event.pos, last_pos,  radius)
                try:
                    x, y = event.pos[1], event.pos[0]
                    arr[x - radius : x + radius, y - radius : y + radius] = 1
                except IndexError:
                    continue
                
                # prediction mechanism
                pygame.draw.rect(screen, (0, 0, 0), (5, 5, 130, 30))
                img = resize(arr, dim, dim)         # final input
                text = predict(img, model)
                render_text(text, 10, 10)
                
            last_pos = event.pos


        pygame.display.flip()

    pygame.quit()
    main(model)      # returing to main menu


# main loop
def main(model):

    # mode of input selection
    mode = input("Select mode of prediction: \nfor FROM CANVAS press 1 \nfor FROM WEBCAM press 2\nand press 0 to exit\n=> ")

    if mode == "1":
        canvas(model)
    elif mode == "2":
        from_cam(model)
    elif mode == "0":
        exit()
    else:
        print("Please enter something valid, lets try this again")
        main(model)


# execution
if __name__ == '__main__': 

    # model importing
    try :
        
        model = load_model('model.h5')      # Modify path if this is not the path to model

    # if model isn't in folder
    except OSError:
        from keras.datasets import mnist
        print("it seems you don't have a model, so we will create a new model, it will take some time")
        epochs = int(input("input no. of epochs (positive integer): "))
        make_model(mnist, epochs)    # predefined model by keras, numbers dataset
        model = load_model('model.h5')

    main(model)
