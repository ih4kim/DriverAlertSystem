# DriverAlertSystem

## System requirements:

```bash
pip install tensorflow
pip install keras 
pip install numpy
pip install pyqt5
pip install imageio
pip install wget
pip install opencv-python
pip install pyserial
```

If you want to see the ML model being trained, take the closed_eye_model file out of the folder and run main.py.


In this project we used Haar cascade in OpenCV to select an ROI around a driver's eyes, then with a dataset of 30,000 images of closed and open eyes collected from http://mrl.cs.vsb.cz/eyedataset, used Tensorflow to classify whether the eyes are open or closed. Then if both eyes are closed for 5 seconds or more, an Arduino spray system starts spraying the driver with water! And this all happens while displaying the driver's eye state and a live stream video of the driver using a webcam and Qt. ...... . . . .  .  .  .   .   .   .    .   
