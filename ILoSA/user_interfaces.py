# input stuff
from pynput.keyboard import Listener, KeyCode
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal
import time

# Virtual method
# This method must be implemented on the inheriting class
# See user_interfaces.py for Keyboard and GUI implementations 
class Base_UI():
    def __init__(self):
        super().__init__()
        return

    def end_demo_user_input(self):
        raise NotImplementedError()

    def save_demo_user_input(self):
        raise NotImplementedError()

class KBUI(Base_UI):
    def __init__(self):
        print('initing keyboard UI')
        super(GUI, self).__init__()
        return

    def end_demo_user_input(self):
        print("Recording started. Press e to stop.")
        self.end = False
        self.listener = Listener(on_press = self._on_press)
        self.listener.start()
        return
    
    def save_demo_user_input(self):
        return input("Do you want to keep this demonstration? [y/n] \n")

    def _on_press(self, key):
        # This function runs on the background and checks if a keyboard key was pressed
        if key == KeyCode.from_char('e'):
            self.end = True
            self.listener.stop()
            return False
        
# signals for the GUI class
# need to be in a class for some reason IDK
class GuiSignals(QObject):
    get_end_demo = pyqtSignal()
    get_save_demo = pyqtSignal()

class GUI(Base_UI):
    def __init__(self):
        print('initing GUI signals')
        super(GUI, self).__init__()
        self.received = None
        self.signals = GuiSignals()
        return

    def end_demo_user_input(self):
        self.end = False
        # ask the main thread to create the "end demo" window
        self.signals.get_end_demo.emit()
    
    def receive(self, data):
        print('panda received %s from main thread'%data)
        self.received = data
        return
                                                               
    def save_demo_user_input(self):
        # Ask the main thread to create the "save demo" window
        self.signals.get_save_demo.emit()
        while self.received == None:
            print("I am waiting....")
            time.sleep(1)
        return self.received
