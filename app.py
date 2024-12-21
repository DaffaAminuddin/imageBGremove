from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import torch
import numpy as np
from torch.autograd import Variable
from skimage import transform
from PIL import Image
from model import U2NET
import cv2


# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load Model
print("---Loading Model---")
current_dir = os.path.dirname(__file__)
model_name = 'u2net'
model_dir = os.path.join(current_dir, 'saved_models', model_name, model_name + '.pth')

net = U2NET(3, 1)
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
net.eval()


# Remove Background Function
def remove_bg(image_path):
    # Load image
    with open(image_path, "rb") as image:
        f = image.read()
        img = bytearray(f)

    nparr = np.frombuffer(img, np.uint8)
    if len(nparr) == 0:
        raise ValueError('Empty image data')

    # Decode and preprocess image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = transform.resize(img, (320, 320), mode='constant')

    tmp_img = np.zeros((img.shape[0], img.shape[1], 3))
    tmp_img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    tmp_img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    tmp_img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

    tmp_img = tmp_img.transpose((2, 0, 1))
    tmp_img = np.expand_dims(tmp_img, 0)
    image = torch.from_numpy(tmp_img).type(torch.FloatTensor)
    image = Variable(image)

    if torch.cuda.is_available():
        image = image.cuda()

    # Perform inference
    d1, *_ = net(image)
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())

    # Convert prediction to image
    predict_np = (pred.squeeze().cpu().data.numpy() * 255).astype(np.uint8)
    im = Image.fromarray(predict_np).convert('L')  # Grayscale image
    original = Image.open(image_path)
    im = im.resize(original.size)  # Resize prediction to match original size

    # Add alpha channel (transparency)
    rgba_image = original.convert('RGBA')
    rgba_image.putalpha(im)

    # Save output
    output_path = "output.png"  # Output file name
    rgba_image.save(output_path, 'PNG')  # Save as PNG to retain transparency

    return output_path  # Return output file path

# API Endpoint
@app.route('/api/remove-background', methods=['POST'])
def api_remove_background():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save uploaded file temporarily
        input_path = os.path.join(current_dir, 'temp_input.png')
        image_file.save(input_path)

        # Process the image
        output_path = remove_bg(input_path)

        # Kirim hasil sebagai response
        return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='output.png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # finally:
    #     # Cleanup temporary files
    #     if os.path.exists(input_path):
    #         os.remove(input_path)
    #     if os.path.exists(output_path):
    #         os.remove(output_path)

# Run the Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)