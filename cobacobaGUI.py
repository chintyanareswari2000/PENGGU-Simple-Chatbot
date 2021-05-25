import tkinter
from tkinter import *

base = Tk()
base.title("PENGGU")
base.geometry("500x500")

SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#ecdfc8", activebackground="#ecb390",fg="#df7861",
                    )

SendButton.place(x=355, y=401, height=90)

base.mainloop()