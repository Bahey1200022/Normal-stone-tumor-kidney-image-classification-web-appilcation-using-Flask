from flask import Flask, render_template,request, jsonify, send_file
import numpy as np
import cv2
import base64
import joblib
mlp = joblib.load('mlp_model.pkl')

app = Flask(__name__)

@app.route('/')
def Stone_model():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    data_url1 = request.json['image_data']
    global img
    pic = []

    img = cv2.imdecode(decodefromjs(data_url1), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128,128))
    # if img.shape[2] ==1:
    #     img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=np.array(img)
    img = img/255
    pic.append(img)
    pic_np = np.array(pic)  # Convert the list pic to a NumPy array
    pic1_2d = pic_np.reshape(pic_np.shape[0], -1)
    global a
    a=mlp.predict(pic1_2d)
    print(a)


    return jsonify({'prediction': a.tolist()})
def decodefromjs(data_url):
        image_data = data_url.split(',')[1]
        

    # Decode the image data from base64
        decoded_data = base64.b64decode(image_data)
    
    # Convert the decoded data to a NumPy array
        np_data = np.frombuffer(decoded_data, np.uint8)
        return np_data

if __name__ == '__main__':
    app.run(debug=True)