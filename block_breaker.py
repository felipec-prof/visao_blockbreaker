import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import math
import random


HANDEDNESS_TEXT_COLOR = (88, 205, 54)
SMOOTH_FACTOR = 0.4          
BALL_SPEED_INIT = 10         
BALL_SPEED_MAX  = 50         
BALL_RADIUS     = 18



square      = [[0.0, 0.0] for _ in range(6)]


def pts(ponto, w, h):
    return (int(square[ponto][0] * w), int(square[ponto][1] * h))


def smooth(new_val, old_val, alpha=SMOOTH_FACTOR):
    return alpha * new_val + (1 - alpha) * old_val


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(min(2, len(hand_landmarks_list))):
        lm = hand_landmarks_list[idx]
        
        if idx == 0:
                square[0][0] = round(lm[4].x,3)
                square[0][1] = round(lm[4].y,3)

                square[1][0] = round(lm[8].x,3)
                square[1][1] = round(lm[8].y,3)
        if idx == 1:
                square[2][0] = round(lm[4].x,3)
                square[2][1] = round(lm[4].y,3)

                square[3][0] = round(lm[8].x,3)
                square[3][1] = round(lm[8].y,3)
    return annotated_image


def exponential_function(channel, exp):
    table = np.array([min(int(i ** exp), 255) for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(channel, table)

def duo_tone(img, exp, s1, s2, s3):
    res = img.copy()
    for i in range(3):
        if i in (s1, s2):
            res[:, :, i] = exponential_function(res[:, :, i], exp)
        else:
            res[:, :, i] = exponential_function(res[:, :, i], 2 - exp) if s3 else 0
    return res


BLOCK_ROWS   = 4
BLOCK_COLS   = 10
BLOCK_H      = 30
BLOCK_PAD    = 5
BLOCK_TOP    = 50       

COLORS_BY_ROW = [
    (60,  60,  220),   
    (60,  160, 220),   
    (60,  220, 220),   
    (60,  200, 80),    
]

def create_blocks(w):
    blocks = []
    total_w = w - 2 * BLOCK_PAD
    bw = (total_w - (BLOCK_COLS - 1) * BLOCK_PAD) // BLOCK_COLS
    for row in range(BLOCK_ROWS):
        for col in range(BLOCK_COLS):
            x = BLOCK_PAD + col * (bw + BLOCK_PAD)
            y = BLOCK_TOP + row * (BLOCK_H + BLOCK_PAD)
            color = COLORS_BY_ROW[row % len(COLORS_BY_ROW)]
            blocks.append({"rect": [x, y, bw, BLOCK_H], "alive": True, "color": color})
    return blocks


def circle_rect_collision(cx, cy, r, rx, ry, rw, rh):
    """Retorna (colidiu, normal_x, normal_y)."""
    nearest_x = max(rx, min(cx, rx + rw))
    nearest_y = max(ry, min(cy, ry + rh))
    dist_x = cx - nearest_x
    dist_y = cy - nearest_y
    dist_sq = dist_x ** 2 + dist_y ** 2
    if dist_sq < r * r:
        dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-6
        return True, dist_x / dist, dist_y / dist
    return False, 0, 0


def check_paddle_collision(cx, cy, r, poly_pts):
    """
    SAT completo: testa todos os eixos das arestas do polígono + eixos ponto→bola.
    Retorna (colidiu, normal_x, normal_y) — normal aponta para FORA do polígono.
    """
    n = len(poly_pts)
    min_overlap = float('inf')
    best_nx, best_ny = 0.0, -1.0

    axes = []
    for i in range(n):
        ax, ay = poly_pts[i]
        bx, by = poly_pts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        length = math.hypot(ex, ey)
        if length < 1e-6:
            continue
        
        axes.append((-ey / length, ex / length))

    for nx, ny in axes:
        
        projs = [nx * vx + ny * vy for vx, vy in poly_pts]
        poly_min, poly_max = min(projs), max(projs)
        
        ball_proj = nx * cx + ny * cy
        ball_min  = ball_proj - r
        ball_max  = ball_proj + r

        
        if ball_max < poly_min or ball_min > poly_max:
            return False, 0.0, 0.0   

        
        overlap = min(ball_max - poly_min, poly_max - ball_min)
        if overlap < min_overlap:
            min_overlap = overlap
            best_nx, best_ny = nx, ny

    
    cx_poly = sum(v[0] for v in poly_pts) / n
    cy_poly = sum(v[1] for v in poly_pts) / n
    to_ball_x = cx - cx_poly
    to_ball_y = cy - cy_poly
    if best_nx * to_ball_x + best_ny * to_ball_y < 0:
        best_nx, best_ny = -best_nx, -best_ny

    return True, best_nx, best_ny


def segment_circle_intersects(ax, ay, bx, by, cx, cy, r):
    """Retorna True se o segmento AB passa dentro do raio r em torno de (cx,cy)."""
    dx, dy = bx - ax, by - ay
    fx, fy = ax - cx, ay - cy
    a = dx*dx + dy*dy
    if a < 1e-12:
        return False
    b = 2 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2*a)
    t2 = (-b + disc) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


particles = []

def spawn_particles(cx, cy, color, n=12):
    for _ in range(n):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 8)
        particles.append({
            "x": cx, "y": cy,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "life": 1.0,
            "color": color,
            "size": random.randint(4, 10),
        })

def update_particles(frame):
    alive = []
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.3          
        p["life"] -= 0.06
        if p["life"] > 0:
            alpha = p["life"]
            color = tuple(int(c * alpha) for c in p["color"])
            cv2.circle(frame, (int(p["x"]), int(p["y"])), p["size"], color, -1)
            alive.append(p)
    particles[:] = alive


def draw_hud(frame, score, lives, level):
    cv2.putText(frame, f"PONTOS: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"VIDAS: {lives}", (frame.shape[1]//2 - 60, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"NIVEL: {level}", (frame.shape[1] - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


camera = cv2.VideoCapture(0)
cv2.namedWindow("Breakout Hands", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Breakout Hands", 1150, 720)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)


ret, frame0 = camera.read()
h, w = frame0.shape[:2]

angle = random.uniform(math.pi / 4, 3 * math.pi / 4)   
circx  = w // 2
circy  = h // 2
speed  = BALL_SPEED_INIT
dx     = speed * math.cos(angle)
dy     = -abs(speed * math.sin(angle))   

score  = 0
lives  = 3
level  = 1
blocks = create_blocks(w)
game_state = "playing" 


while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)
    annotated = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    frame_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    
    p0 = pts(0, w, h)
    p1 = pts(1, w, h)
    p2 = pts(2, w, h)
    p3 = pts(3, w, h)

    paddle_poly = np.array([p0, p1, p3, p2], np.int32)

    
    mask1 = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask1, [paddle_poly.reshape((-1, 1, 2))], 255)
    imgduo = duo_tone(frame_bgr, 1.1, 2, 3, 0)
    frame_bgr[mask1 == 255] = imgduo[mask1 == 255]

    
    cv2.polylines(frame_bgr, [paddle_poly.reshape((-1, 1, 2))], True, (0, 0, 200), 2)

    
    if game_state == "playing":

        poly_list = [tuple(p) for p in paddle_poly.tolist()]

        
        
        
        prev_cx, prev_cy = circx, circy
        swept_hit = False
        for i in range(len(poly_list)):
            ax, ay = poly_list[i]
            bx, by = poly_list[(i + 1) % len(poly_list)]
            if segment_circle_intersects(prev_cx, prev_cy,
                                         prev_cx + dx, prev_cy + dy,
                                         (ax + bx) / 2, (ay + by) / 2,
                                         BALL_RADIUS + math.hypot(bx-ax, by-ay)/2):
                swept_hit = True
                break

        
        steps = max(1, int(math.hypot(dx, dy) / (BALL_RADIUS * 0.5)))
        sdx, sdy = dx / steps, dy / steps
        collided_paddle = False
        lost_life = False

        for _ in range(steps):
            circx += sdx
            circy += sdy

            
            if circx - BALL_RADIUS <= 0:
                circx = BALL_RADIUS
                dx = abs(dx);  sdx = dx / steps
            elif circx + BALL_RADIUS >= w:
                circx = w - BALL_RADIUS
                dx = -abs(dx); sdx = dx / steps

            
            if circy - BALL_RADIUS <= 0:
                circy = BALL_RADIUS
                dy = abs(dy);  sdy = dy / steps

            
            if circy + BALL_RADIUS >= h:
                lives -= 1
                circx, circy = w // 2, h // 2
                angle = random.uniform(math.pi / 4, 3 * math.pi / 4)
                dx = speed * math.cos(angle)
                dy = -abs(speed * math.sin(angle))
                if lives <= 0:
                    game_state = "lose"
                lost_life = True
                break

            
            if not collided_paddle:
                hit, nx, ny = check_paddle_collision(circx, circy, BALL_RADIUS, poly_list)

                
                if not hit and swept_hit:
                    hit, nx, ny = check_paddle_collision(
                        circx, circy, BALL_RADIUS + 6, poly_list)

                if hit:
                    
                    circx += nx * (BALL_RADIUS + 2)
                    circy += ny * (BALL_RADIUS + 2)

                    
                    dot = dx * nx + dy * ny
                    dx -= 2 * dot * nx
                    dy -= 2 * dot * ny

                    
                    speed = min(math.hypot(dx, dy) + 0.3, BALL_SPEED_MAX)
                    norm = math.hypot(dx, dy)
                    dx, dy = dx / norm * speed, dy / norm * speed
                    sdx, sdy = dx / steps, dy / steps
                    collided_paddle = True

            
            for block in blocks:
                if not block["alive"]:
                    continue
                rx, ry, rw, rh = block["rect"]
                hit, nx, ny = circle_rect_collision(
                    int(circx), int(circy), BALL_RADIUS, rx, ry, rw, rh
                )
                if hit:
                    block["alive"] = False
                    score += 10 * level
                    spawn_particles(int(circx), int(circy), block["color"])
                    dot = dx * nx + dy * ny
                    dx -= 2 * dot * nx
                    dy -= 2 * dot * ny
                    sdx, sdy = dx / steps, dy / steps
                    break

        
        if not lost_life and all(not b["alive"] for b in blocks):
            level += 1
            speed = min(BALL_SPEED_INIT + level * 1.5, BALL_SPEED_MAX)
            blocks = create_blocks(w)
            game_state = "level_clear"

    
    for block in blocks:
        if not block["alive"]:
            continue
        rx, ry, rw, rh = block["rect"]
        cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), block["color"], -1)
        cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1)

    
    update_particles(frame_bgr)

    
    cv2.circle(frame_bgr, (int(circx), int(circy)), BALL_RADIUS, (255, 60, 20), -1)
    cv2.circle(frame_bgr, (int(circx), int(circy)), BALL_RADIUS, (255, 200, 100), 2)

    
    frame_bgr = cv2.flip(frame_bgr, 1)
    draw_hud(frame_bgr, score, lives, level)

    
    if game_state == "lose":
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)
        cv2.putText(frame_bgr, "GAME OVER", (w//4 + 40, h//2 - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 220), 3)
        cv2.putText(frame_bgr, f"Pontos: {score}   [R] reiniciar  [ESC] sair",
                    (w//4 + 10, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    elif game_state == "level_clear":
        cv2.putText(frame_bgr, f"NIVEL {level - 1} COMPLETO!  Preparando...",
                    (w//4, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.2, (60, 230, 60), 3)
        game_state = "playing"   
    
    cv2.imshow("Breakout Hands", frame_bgr)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:       
        break
    elif key == ord('r') or key == ord('R'):   
        score, lives, level = 0, 3, 1
        speed = BALL_SPEED_INIT
        angle = random.uniform(math.pi / 4, 3 * math.pi / 4)
        circx, circy = w // 2, h // 2
        dx = speed * math.cos(angle)
        dy = -abs(speed * math.sin(angle))
        blocks = create_blocks(w)
        particles.clear()
        game_state = "playing"

camera.release()
cv2.destroyAllWindows()