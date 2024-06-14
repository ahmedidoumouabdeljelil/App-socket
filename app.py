import threading
import time
import os
import pyrebase
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET')
socketio = SocketIO(app, cors_allowed_origins="*")

firebaseConfig = {
    "apiKey": os.environ.get('FIREBASE_API_KEY'),
    "authDomain": os.environ.get('FIREBASE_AUTH_DOMAIN'),
    "databaseURL": os.environ.get('FIREBASE_DATABASE_URL'),
    "projectId": os.environ.get('FIREBASE_PROJECT_ID'),
    "storageBucket": os.environ.get('FIREBASE_STORAGE_BUCKET'),
    "messagingSenderId": os.environ.get('FIREBASE_MESSAGING_SENDER_ID'),
    "appId": os.environ.get('FIREBASE_APP_ID'),
    "measurementId": os.environ.get('FIREBASE_MEASUREMENT_ID')
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

# Load the TFLite model
interpreter = tflite.Interpreter(model_path='model_GRU_3.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_data_and_predict():
    while True:
        try:
            data = db.get().val()
            if isinstance(data, dict):
                courant = data.get('Courant', 0)
                tension = data.get('Tension', 0)
                temperature = data.get('Temperature', 0)
                input_data = np.array([[courant, tension, temperature]], dtype=np.float32)
                
                # Run the TFLite model
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                soc = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # Emit the SocketIO event with the prediction
                socketio.emit('prediction', {
                    'Courant': courant,
                    'Tension': tension,
                    'Temperature': temperature,
                    'SOC': soc.tolist()
                })
            else:
                print("Les données récupérées ne sont pas au format attendu :", data)
        except Exception as e:
            print("Erreur lors de la récupération des données ou de la prédiction :", str(e))
        
        time.sleep(5)

@app.route('/data')
def get_data():
    try:
        # Récupérez les données de la base de données Firebase
        data = db.get().val()
        
        # Effectuez votre prédiction à partir des données
        courant = data.get('Courant', 0)
        tension = data.get('Tension', 0)
        temperature = data.get('Temperature', 0)
        input_data = np.array([[courant, tension, temperature]], dtype=np.float32)
        
        # Exécutez le modèle TFLite pour obtenir la prédiction
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        soc_prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Retournez les données de la base de données et la prédiction sous forme de réponse JSON
        return jsonify({'Courant': courant, 'Tension': tension, 'Temperature': temperature, 'SOC_Prediction': soc_prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    threading.Thread(target=load_data_and_predict).start()
    socketio.run(app, debug=True)

