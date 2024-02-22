import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show(data, index):  # currently uncalled

    plt.imshow(data[index], cmap=plt.get_cmap('gray'))  
    plt.show()          #shows image as a graph
    print(data[index])


def make_model(data):

    (x_train, y_train), (x_test, y_test) = data.load_data()         # splitting data into x, y and training and testing
    x_train, x_test = x_train / 255, x_test / 255       # normalistation

    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # image into 28*28 = 784 nodes
    keras.layers.Dense(128, activation='relu'),     # hidden layer of 128 nodes
    keras.layers.Dropout(0.2),                      # dropout layer to prevent overfitting
    keras.layers.Dense(10, activation='softmax')    # output layer of 10 nodes for 10 classes
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    model.fit(x_train, y_train, epochs=5)       # training the model

    model.evaluate(x_test, y_test)      # evaluation

    model.save('model.h5')      # saving to .h5

def main():
    try :
        model = keras.models.load_model('model.h5')

    except OSError:
        make_model(keras.datasets.mnist)    # predefined model by keras, numbers dataset
        model = keras.models.load_model('model.h5')

    output = ""     # prediction
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
        cv2.putText(img, str(output), (250, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)  # prediction

        cv2.rectangle(img, (250, 170), (390, 310), (0, 255, 0), 3)

        cv2.imshow("input", cv2.resize(input_img, (280, 280)))  # model input
        cv2.imshow("cam", img)  # actual camera feed
    
        key = cv2.waitKey(0)
        if key == ord('x'): # exit
            break 
        if key == ord('b'): # toggle bot
            on = True if not on else False
            
        if on:
            try:
                predicted_probabilities = model.predict(np.expand_dims(input_img, axis=0))  # gives probability of every class
                pred = predicted_probabilities.argmax()     # finds most probable prediction
                percentage = int(predicted_probabilities[0, pred] * 10000) / 100  # probability of that prediction
                output = f"{pred} : {percentage}%"      # prediction
                print(output)
    
            except:
                print("failed")
                break

    cam.release()
    cv2.destroyAllWindows()     # removing screen

    
if __name__ == '__main__':
    main()
