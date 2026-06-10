import cv2
import numpy as np
import os

# Outros Sensores
SENSORES = {
    "TEMPERATURA":  {"valor": 33,    "unidade": "C",   "min": 18,  "max": 30},
    "UMIDADE":      {"valor": 55,    "unidade": "%",   "min": 60,  "max": 85},
    "CO2":          {"valor": 1200,  "unidade": "ppm", "min": 400, "max": 1500},
    "O2":           {"valor": 19,    "unidade": "%",   "min": 18,"max": 23},
    "RADIACAO":     {"valor": 0.15,  "unidade": "mSv", "min": 0,   "max": 1},
    "PH_AGUA":      {"valor": 6.5,   "unidade": "pH",  "min": 5.5, "max": 7.0},
    "LUMINOSIDADE": {"valor": 800,   "unidade": "lux", "min": 500, "max": 10000},
}

def avaliar_sensor(dados):
    v, mn, mx = dados["valor"], dados["min"], dados["max"]
    if v < mn:
        return "BAIXO", (0, 100, 255)
    elif v > mx:
        return "ALTO", (0, 0, 255)
    else:
        return "OK", (0, 220, 0)

# IMG
script_dir = os.path.dirname(os.path.abspath(__file__))
frame = cv2.imread(os.path.join(script_dir, "image9.png"))

if frame is None:
    print("Erro ao abrir a imagem.")
    exit()

frame = cv2.resize(frame, (640, 480))


hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Vermelho
mask_vermelho1 = cv2.inRange(hsv, np.array([0,  80, 50]), np.array([10, 255, 255]))
mask_vermelho2 = cv2.inRange(hsv, np.array([165, 80, 50]), np.array([180, 255, 255]))
mask_vermelho  = cv2.bitwise_or(mask_vermelho1, mask_vermelho2)

# Laranja
mask_laranja = cv2.inRange(hsv, np.array([11, 80, 50]), np.array([25, 255, 255]))

# Verde
mask_verde = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

# Máscara
mask = cv2.bitwise_or(mask_vermelho, cv2.bitwise_or(mask_laranja, mask_verde))

total_pixels   = frame.shape[0] * frame.shape[1]
pct_vermelho   = cv2.countNonZero(mask_vermelho) / total_pixels * 100
pct_laranja    = cv2.countNonZero(mask_laranja)  / total_pixels * 100
pct_verde      = cv2.countNonZero(mask_verde)    / total_pixels * 100

# Ponto de colheita
if pct_vermelho >= 5:
    status_cor = "PRONTO PARA COLHEITA"
    color_cor  = (0, 0, 255)
elif pct_laranja >= 5:
    status_cor = "PINTADO / QUASE PRONTO"
    color_cor  = (0, 140, 255)
else:
    status_cor = "VERDE / NAO COLHER"
    color_cor  = (0, 200, 0)

cv2.putText(frame, f"Vermelho:{pct_vermelho:.1f}% Laranja:{pct_laranja:.1f}% Verde:{pct_verde:.1f}%",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
cv2.putText(frame, f"Colheita: {status_cor}", (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color_cor, 2)

# Detecta Tomates
for msk, cor_cnt in [(mask_vermelho, (0, 0, 255)),
                     (mask_laranja,  (0, 140, 255)),
                     (mask_verde,    (0, 200, 0))]:
    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor_cnt, 2)

# Painel
painel_x = 0
painel_y = 90
linha_altura = 38

overlay = frame.copy()
cv2.rectangle(overlay, (painel_x, painel_y - 5),
              (380, painel_y + linha_altura * len(SENSORES) + 5), (30, 30, 30), -1)
cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


# Alertas
alertas = []

for i, (nome, dados) in enumerate(SENSORES.items()):
    estado, cor_estado = avaliar_sensor(dados)
    v   = dados["valor"]
    uni = dados["unidade"]
    mn  = dados["min"]
    mx  = dados["max"]

    y_pos = painel_y + i * linha_altura + 25

    # Nome e valor
    cv2.putText(frame, f"{nome}: {v} {uni}", (painel_x + 8, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # Estado
    cv2.putText(frame, f"[{estado}]", (painel_x + 270, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, cor_estado, 2)

    if estado != "OK":
        alertas.append((nome, estado, v, uni, mn, mx))

# Alertas
if alertas:
    print("\n===== ALERTAS =====")
    for nome, estado, v, uni, mn, mx in alertas:
        if estado == "BAIXO":
            print(f"[BAIXO]  {nome}: {v} {uni}  (minimo esperado: {mn} {uni})")
        else:
            print(f"[ALTO]   {nome}: {v} {uni}  (maximo esperado: {mx} {uni})")
    print("===================\n")
else:
    print("Todos os sensores dentro dos limites normais.")

# Img
cv2.imshow("AgroMars Vision", frame)
cv2.imshow("Mascara Tomate", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
