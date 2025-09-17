import cv2
import os
import datetime
import time
import numpy as np
from tensorflow import keras
from keras_preprocessing.image import img_to_array


sentence = ' '
start_time = time.time()


def generate_sentence(img, prediction, probability, interval=5):
	'''this is a function that tries to capture different sentences in real time, 
	it's just a simple way to try to display the composition of a sentence in real time on the terminal.'''
	global sentence
	global start_time 
	if probability >= 0.85  and sentence[-1] != prediction:
		sentence += str(prediction)
		print(sentence)
		start_time = time.time()
	elif probability < 0.85 and (time.time() - start_time > interval) and sentence[-1] != ' ':
		sentence += ' '
		print('...start new word...')
		start_time = time.time()
	elif (time.time() - start_time > interval * 2):
		sentence = ' '
		print(sentence)
		print('...end of sentence...')
		start_time = time.time()



def main(window_name = "frame", delay=1):
    model_loaded = keras.models.load_model('CNN.h5') # Load the trainded model (CNN)
    # This following list of letters is used to map the predictions calculated by the trained model (CNN) 
    # in the correct order with respect to the String Indexer used in the data preprocessing phase before training the model.
    alphabet = ['A', 'B', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'C', 'V', 'W', 'X', 'Y', 'D', 'E', 'F', 'G', 'H', 'I', 'K'] 

    cap = cv2.VideoCapture(0) #Open the laptop video camera if it exists

    if not cap.isOpened():
        return

    idx = 0
    num_images = 0
    past_time = time.time()
    while True:
        ret, frame = cap.read()
        rect = cv2.rectangle(frame, (40, 40), (600, 600), (0, 255, 0), 2) #Generate a rectangle within which we have to place our hand so that the sign-letter represented by the gesture I make with my hand is predicted.
        image = frame
        image = image[40:600, 40:600] # I only consider the image within the green rectangle previously built in the upper left corner.
        image = cv2.resize(image, (256, 256))
        # In the next seven rows I apply the Color thresholding strategy to highlight only the hand in white and make the background black.
        result = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0,60,110])
        upper = np.array([255,209,255]) 
        mask = cv2.inRange(image, lower, upper)
        result = cv2.bitwise_and(result, result, mask=mask)

        mask = cv2.resize(mask, (28, 28)) # Then crop the image within the rectangle to a size of 28 x 28 because the images that the Convolutional Neural Network was trained on are 28 x 28 in grayscale.
        img_array = np.array(mask)
        img_array = img_array.astype("float") / 255.0 # Scale the captured image by dividing by 255
        img_array = np.expand_dims(img_array, axis=0) #Expand dimensions to match the 4D Tensor shape.
        img_array= img_array.reshape(-1,784)  
        predictions_prob = model_loaded.predict(img_array) # Bring out the different classification probabilities for that image. What are the chances that the image captured by the camera represents each of the 24 letters of the ASL alphabet?
        prob = max(predictions_prob[0]) 
        idx = np.argmax(predictions_prob[0]) # Select the prediction with the highest probability.
        pred = alphabet[idx] # Map this prediction to the corresponding letter of the alphabet.

        current_time = time.time()
        if (current_time - past_time) > 1:
        	generate_sentence(img_array, pred, prob)
        	past_time = current_time

        prediction = "Predicion: {}, with probability: {}".format(pred, prob) 
        cv2.putText(rect, prediction, (40, 40-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) # Show the prediction and the probability with which this prediction was made on the upper left edge of the green rectangle box.
        cv2.imshow(window_name, frame)
        cv2.imshow('mask', mask)
        frame = mask        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
