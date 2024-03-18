import cv2
import numpy as np

def detection_couleur_dans_contour(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 1000)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Extraire la région d'intérêt (ROI) basée sur le contour
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]

        # Convertir l'ROI en espace de couleur HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Plages de couleurs
        lower_rouge = np.array([0, 100, 100])
        upper_rouge = np.array([10, 255, 255])
        lower_vert = np.array([40, 40, 40])
        upper_vert = np.array([80, 255, 255])
        lower_bleu = np.array([110, 50, 50])
        upper_bleu = np.array([130, 255, 255])
        lower_jaune = np.array([20, 100, 100])
        upper_jaune = np.array([30, 255, 255])

        # Créer les masques pour chaque plage de couleur
        mask_red = cv2.inRange(hsv_roi, lower_rouge, upper_rouge)
        mask_vert = cv2.inRange(hsv_roi, lower_vert, upper_vert)
        mask_bleu = cv2.inRange(hsv_roi, lower_bleu, upper_bleu)
        mask_jaune = cv2.inRange(hsv_roi, lower_jaune, upper_jaune)

        # Compter le nombre de pixels dans chaque plage de couleur
        pixel_count_red = cv2.countNonZero(mask_red)
        pixel_count_vert = cv2.countNonZero(mask_vert)
        pixel_count_bleu = cv2.countNonZero(mask_bleu)
        pixel_count_jaune = cv2.countNonZero(mask_jaune)

        # Déterminer la couleur prédominante pour chaque ROI
        max_pixel_count = max(pixel_count_red, pixel_count_vert, pixel_count_bleu, pixel_count_jaune)

        if max_pixel_count == pixel_count_red:
            couleur = "rouge"
        elif max_pixel_count == pixel_count_vert:
            couleur = "vert"
        elif max_pixel_count == pixel_count_bleu:
            couleur = "bleu"
        elif max_pixel_count == pixel_count_jaune:
            couleur = "jaune"
        else:
            couleur = "Couleur non reconnue"

        
        cv2.drawContours(image, [contour], -1, (0), 5)
        cv2.putText(image, couleur, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

    cv2.imshow('Contours et Couleurs', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Exemple d'utilisation
detection_couleur_dans_contour('/Users/mathistelle/IATIC/myenv/Nyrio/visionsep.jpeg')
