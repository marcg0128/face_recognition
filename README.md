# face_recognition

## Instalation
```
pip install -r requirements.txt
```
or
```
py -m pip install -r requirements.txt
```

## Start
1. Ensure that the application has write access to the current directory.
2. The application uses a local SQLite database stored in `./database.sqlite`. No manual schema creation is required, as the database will be initialized automatically when the application runs.
3. Plug a webcam in your device
4. Start the code with:

   ```
   py main.py
   ```

## Usage
Press: <br>
`s` - saving the face (**IMPORTANT!!! Currently you only can save a face when only one is visible**) <br>
`q` - quitting the webcam
