import cv2 as cv
import numpy as np
import queue
from time import time
from sklearn.linear_model import LinearRegression
from config import *

class VisionSystem:
    def __init__(self):
        self.vidCap = cv.VideoCapture(0)
        self.vidCap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.positions = queue.Queue(N_Positions)
        self.old_ball = None
        self.old_time = time()

    def predict(self, points):
        if len(points) > 1:
            X = np.array([p[1] for p in points]).reshape((-1,1))
            Y = np.array([p[0] for p in points])
            reg = LinearRegression().fit(X, Y)
            x_pred = round(reg.intercept_ + reg.coef_[0] * yArr)

            x_pred = min(max(x_pred, xLimRight), xLimLeft)
            return [x_pred, yArr]
        return None

    def get_frame_and_prediction(self):
        ret, frame = self.vidCap.read()
        if not ret:
            return None, None, None

        # Convertir l'image en espace de couleur HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Définir les plages de couleurs pour détecter la balle blanche
        # Balle blanche : une plage de teintes (H) et des valeurs de saturation (S) et luminosité (V) élevées
        lower_white = np.array([0, 0, 180])  # Plage basse du blanc
        upper_white = np.array([255, 60, 255])  # Plage haute du blanc

        # Appliquer un masque pour extraire la couleur blanche
        mask = cv.inRange(hsv, lower_white, upper_white)

        # Appliquer une opération de nettoyage : ouverture (dilater puis eroder)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5,5), np.uint8))

        # Trouver les contours de la balle dans l'image
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        shoot = 0
        point_final = None
        ball = None

        # Chercher les contours les plus grands (qui devraient être la balle)
        for contour in contours:
            if cv.contourArea(contour) > 500:  # Filtrer les petites zones
                (x, y), radius = cv.minEnclosingCircle(contour)
                if radius > 10:
                    ball = (int(x), int(y))
                    cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)  # Dessiner un cercle vert

                    if self.old_ball is not None:
                        to_goal = ball[1] < self.old_ball[1]
                        dist = abs(ball[0] - self.old_ball[0]) + abs(ball[1] - self.old_ball[1])
                        if dist >= deltaMove:
                            if self.positions.full():
                                self.positions.get()
                            self.positions.put(ball)

                            if to_goal:
                                point_final = self.predict(list(self.positions.queue))
                                if point_final:
                                    delta_time = time() - self.old_time
                                    if delta_time > 0:
                                        vitesse = dist / delta_time
                                        dist_final = abs(ball[0] - point_final[0]) + abs(ball[1] - point_final[1])
                                        t_final = dist_final / vitesse * 1000
                                        if t_final < timeToShoot:
                                            shoot = 1

                    self.old_ball = ball
                    self.old_time = time()

        return frame, point_final, shoot

    def release(self):
        self.vidCap.release()

# Utilisation avec caméra
vision_system = VisionSystem()

while True:
    frame, prediction, shoot = vision_system.get_frame_and_prediction()

    if frame is None:
        break

    # Affichage des résultats dans la fenêtre
    if prediction:
        cv.circle(frame, (prediction[0], prediction[1]), 10, (0, 0, 255), 2)
    
    if shoot:
        cv.putText(frame, 'Shoot!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

    cv.imshow('Frame', frame)

    # Appuyer sur 'q' pour quitter
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vision_system.release()
cv.destroyAllWindows()
