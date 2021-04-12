from tkinter import *

import os
import cv2
import numpy as np
from keras.models import load_model

Dictionary = {0: 'Airplane', 1: 'Automoblie', 2: 'Bird', 3: 'Cat', 4: 'Deer',
         5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'}

model = load_model('D:/pycharm_pro/Project/graph_recog/trained_model/vgg/model_vgg_adadelta_b128_e100.h5')

class Reg(Frame):
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        self.path = Label(frame, text='path')
        self.path.grid(row=0, column=0, sticky=W)
        self.ent = Entry(frame)
        self.ent.grid(row=0, column=1, sticky=W)
        self.submit = Button(frame, text='Submit', command=self.Prediction)
        self.submit.grid(row=1, column=1, sticky=W)
        self.result = Label(frame, text='')
        self.result.grid(row=2, column=0, sticky=W)
    def Prediction(self):
        s = self.ent.get()
        if not os.path.exists(s):
            self.label3['text'] = '{} does not exist'
        else:
            img_pro = cv2.imread(s)
            img_resize = cv2.resize(img_pro, (32, 32))
            img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            img = img.reshape(1, 32, 32, 3)
            predict = np.argmax(model.predict(img))
            self.result['text'] = 'Recognized as : {}'.format(Dictionary[predict])
        self.ent.delete(0, len(s))

root = Tk()
root.geometry('700x300')
root.title('Graph Recognition')
app = Reg(root)

if __name__ == '__main__':
    root.mainloop()