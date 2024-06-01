import dlib
import cv2

# Carica il modello pre-addestrato per il riconoscimento del volto
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Assicurati di scaricare il file del predictor dei landmark del viso

# Carica il video da analizzare
video_capture = cv2.VideoCapture(0)  # Se vuoi usare la webcam, passa 0. Altrimenti, passa il percorso del file video.

while True:
    # Leggi il frame corrente dal video
    ret, frame = video_capture.read()
    
    # Verifica se il frame è stato letto correttamente
    if not ret:
        print("Errore nella cattura del frame")
        break
    
    # Converti il frame in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Rileva i volti nell'immagine
    faces = detector(gray)
    
    # Per ogni volto rilevato
    for face in faces:
        # Ottieni le coordinate del rettangolo del volto
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        # Disegna un rettangolo verde intorno al volto
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Ottieni i landmark facciali
        landmarks = predictor(gray, face)
        
        # Disegna i punti dei landmark sul volto
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        # Collega i punti dei landmark con linee
        for i in range(1, 68):
            cv2.line(frame, (landmarks.part(i - 1).x, landmarks.part(i - 1).y), (landmarks.part(i).x, landmarks.part(i).y), (255, 255, 255), 1)
    
    # Visualizza il frame risultante
    cv2.imshow('Video', frame)
    
    # Interrompi il ciclo se viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la cattura video e chiudi la finestra
video_capture.release()
cv2.destroyAllWindows()
