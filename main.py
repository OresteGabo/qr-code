import cv2
import qrcode
from scipy.spatial import Voronoi, voronoi_plot_2d
# from qrcode import QRCode, constants
import pyqrcode
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def generate_qr_code(data: str, file_name: str):
    qr = pyqrcode.create(data)
    qr.png(file_name, scale=6)


def extract_qr_code(file_name: str) -> str:
    qr_code_detector = cv2.QRCodeDetector()

    img = cv2.imread(file_name)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, decoded_info, points, straight_qrcode = qr_code_detector.detectAndDecodeMulti(img)

    if decoded_info:
        return decoded_info[0]
    else:
        return ''


def hide_data(main_image: str, data_image: str, encrypted_image: str):
    # Load the images
    main = cv2.imread(main_image)
    data = cv2.imread(data_image)

    # Ensure the data_image can be embedded in main_image
    if main.shape[0] < data.shape[0] or main.shape[1] < data.shape[1]:
        raise ValueError("The main image is not large enough to embed the data image")

    # Embed the data image in the main image
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            for channel in range(data.shape[2]):
                main[row, col, channel] = (main[row, col, channel] & 0xFE) | (data[row, col, channel] >> 7)

    cv2.imwrite(encrypted_image, main)


def extract_data(encrypted_image: str, data_image_shape: tuple):
    # Load the encrypted image
    encrypted = cv2.imread(encrypted_image)

    # Create a new image by taking the LSB from the encrypted image
    data_extracted = np.zeros(data_image_shape, dtype=np.uint8)
    for row in range(data_image_shape[0]):
        for col in range(data_image_shape[1]):
            for channel in range(data_image_shape[2]):
                data_extracted[row, col, channel] = (encrypted[row, col, channel] & 1) << 7

    return data_extracted


def generate_voronoi(points: list):
    vor = Voronoi(points)
    return vor


def show_voronoi(vor):
    voronoi_plot_2d(vor)
    plt.show()


from PIL import Image


def generate_blank_image(width: int, height: int, filename: str):
    img = Image.new('RGB', (width, height), color='white')
    img.save(filename)


def main():
    # Générer le code QR
    data = "Hello, World!"
    qr_file = 'qr.png'
    generate_qr_code(data, qr_file)

    # Cacher le code QR dans une image principale
    main_image = 'main_image.png'  # Fichier de votre image principale
    encrypted_image = 'encrypted.png'  # Le nom du fichier dans lequel sauvegarder l'image modifiée
    hide_data(main_image, qr_file, encrypted_image)

    # Extraire le code QR de l'image
    extracted_qr = extract_qr_code(encrypted_image)
    print(f'The extracted QR code data is: {extracted_qr}')


# Generate a blank main image
generate_blank_image(600, 400, 'main_image.png')


main()
