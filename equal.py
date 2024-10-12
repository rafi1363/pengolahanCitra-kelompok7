import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class ImageEqualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Contrast Enhancement")
        
        self.upload_button = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, path):
        # Membaca citra
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # Melakukan histogram equalization
        equalized_image = cv2.equalizeHist(image)

        # Menampilkan citra dan histogram
        self.show_images(image, equalized_image)

    def show_images(self, original, equalized):
        # Mengubah citra ke format yang bisa ditampilkan di Tkinter
        original_image = Image.fromarray(original)
        equalized_image = Image.fromarray(equalized)

        # Konversi ke PhotoImage untuk Tkinter
        self.original_photo = ImageTk.PhotoImage(original_image)
        self.equalized_photo = ImageTk.PhotoImage(equalized_image)

        # Membuat jendela baru untuk menampilkan gambar dan histogram
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        # Menampilkan citra asli
        axs[0, 0].imshow(original, cmap='gray')
        axs[0, 0].set_title('Citra Asli')
        axs[0, 0].axis('off')

        # Menampilkan histogram citra asli
        axs[0, 1].hist(original.ravel(), bins=256, range=[0, 256], color='gray')
        axs[0, 1].set_title('Histogram Citra Asli')
        axs[0, 1].set_xlim([0, 256])
        
        # Menampilkan citra setelah equalization
        axs[1, 0].imshow(equalized, cmap='gray')
        axs[1, 0].set_title('Citra Setelah Equalization')
        axs[1, 0].axis('off')

        # Menampilkan histogram citra setelah equalization
        axs[1, 1].hist(equalized.ravel(), bins=256, range=[0, 256], color='gray')
        axs[1, 1].set_title('Histogram Citra Setelah Equalization')
        axs[1, 1].set_xlim([0, 256])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = Tk()
    app = ImageEqualizationApp(root)
    root.mainloop()