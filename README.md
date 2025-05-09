Info:
1. main.py contains both the ambulance AND passenger speech detection
2. passenger.py contains only the passenger speech detection
3. ambulance ML model uses tenserflow yamnet. speech detection model uses vosk

the passenger speech detection is bad because you have to yell. 

How to run:

make sure you have python 3 installed
```python --version```

enter the virtual python environment where you will run it
```source venv/bin/activate```

install packages
```pip install -r requirements.txt```

run the file
```python main.py```

if you get an error that a package isnt installed, install it:
```pip install some-package```
package names are at the top of the main.py file
