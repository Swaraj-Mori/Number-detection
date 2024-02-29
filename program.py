import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pygame, sys


# if ou want it shows image in form of array (currently uncalled)
def show(data, index):

    plt.imshow(data[index], cmap=plt.get_cmap('gray'))  
    plt.show()          #shows image as a graph
    print(data[index])


# function to make a neural network
def make_model(data, epochs):

    (x_train, y_train), (x_test, y_test) = data.load_data()         # splitting data into x, y and training and testing
    x_train, x_test = x_train / 255, x_test / 255       # normalistation

    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # image into 28*28 = 784 nodes
    keras.layers.Dense(128, activation='relu'),     # hidden layer of 128 nodes
    keras.layers.Dropout(0.2),                      # dropout layer to prevent overfitting
    keras.layers.Dense(10, activation='softmax')    # output layer of 10 nodes for 10 classes
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    model.fit(x_train, y_train, epochs=epochs)       # training the model

    model.evaluate(x_test, y_test)      # evaluation

    model.save('model.h5')      # saving to .h5


# prediction function
def predict(input_image, ml_model):
    predicted_probabilities = ml_model.predict(np.expand_dims(input_image, axis=0))  # gives probability of every class
    pred = predicted_probabilities.argmax()     # finds most probable prediction
    percentage = int(predicted_probabilities[0, pred] * 10000) / 100  # probability of that prediction
    output = f"{pred} : {percentage}%"      # prediction
    return str(output)


# taking from camera
def from_cam(model):
    prediction = "" # prediction variable
    on = False      # checks if prediciton is on or not
    cam = cv2.VideoCapture(0)

    print("Instructions:\n1)Use light background and dark pen an provide adequade light\n2)press b to toggle bot (!fps drop)\n3)press x to exit")

    while True:
        _, img = cam.read() #img stores camera feed

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # grayscale image
        input_img = input_img[170:310, 250:390]         # cropping
        input_img = cv2.threshold(input_img, 125, 255, cv2.THRESH_BINARY_INV)[1]  #thresholding
        input_img = cv2.resize(input_img, (28, 28))     # final input image
    
        text = "Bot on" if on else "Bot off"

        cv2.putText(img, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)    # prediction state
        cv2.putText(img, prediction, (250, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)  # displaying prediction

        cv2.rectangle(img, (250, 170), (390, 310), (0, 255, 0), 3)

        cv2.imshow("input", cv2.resize(input_img, (280, 280)))  # model input
        cv2.imshow("cam", img)  # actual camera feed
    
        key = cv2.waitKey(1)  # checks key if pressed

        if key == ord('x'): # exit
            break 
        if key == ord('b'): # toggle bot
            on = True if not on else False
            
        # prediction mechanism
        if on:
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
    pygame.font.init()

    dim = 28    # shape of the drawn input
    screen = pygame.display.set_mode((dim*10, dim*10))
    pygame.display.set_caption('canvas')
    
    
    draw_on = False
    last_pos = (0, 0)
    arr = np.zeros((dim, dim))  # array wihch stores the pixels, as an image

    radius = 5
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
        
        if event.type == pygame.MOUSEBUTTONUP:          # user is not drawing not drawing
            draw_on = False
        
        # drawing mechanism
        if event.type == pygame.MOUSEMOTION:            
            if draw_on:
                pygame.draw.circle(screen, white, event.pos, radius)
                roundline(screen, white, event.pos, last_pos,  radius)
                try:
                    arr[event.pos[1]//10, event.pos[0]//10] = 1
                except IndexError:
                    continue
            last_pos = event.pos

        # prediction mechanism (it only predicts once you stop drawing)
        elif not draw_on:
            pygame.draw.rect(screen, (0, 0, 0), (5, 5, 130, 30))
            text = predict(arr, model)
            render_text(text, 10, 10)

        pygame.display.flip()

    pygame.quit()
    main(model)      # returing to main menu


# main loop
def main(model):

    # mode of input selection
    mode = input("Select mode of prediction: \nfor FROM CANVAS press 1 \nfor FROM WEBCAM press 2\nand press 0 to exit\n=>")

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
        model = keras.models.load_model('model.h5')

    # if model isn't in folder
    except OSError:
        print("it seems you don't have a model, so we will create a new model, it will take some time")
        epochs = int(input("input no. of epochs (positive integer): "))
        make_model(keras.datasets.mnist, epochs)    # predefined model by keras, numbers dataset
        model = keras.models.load_model('model.h5')

    main(model)
