
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pygame


WIDTH, HEIGHT, FPS = 1100, 680, 60
# OpenCV hue is 0..179. This band covers the user's violet/purple fabric
# while the saturation floor rejects skin, white walls, and most background.
PURPLE_GLOVE_LOW = np.array([112, 48, 38], dtype=np.uint8)
PURPLE_GLOVE_HIGH = np.array([162, 255, 255], dtype=np.uint8)


class MotionTracker:


    def __init__(self):
        self.previous = [None, None]
        self.previous_time = [None, None]
        self.smooth_centers = [None, None]
        self.last_punch = [0.0, 0.0]
        self.last_dodge = 0.0
        self.hands_were_joined = False
        self.kernel = np.ones((5, 5), np.uint8)

    def process(self, frame):
        h, w = frame.shape[:2]
        softened = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(softened, cv2.COLOR_BGR2HSV)
        # Purple fabric occupies one continuous hue band in HSV.
        mask = cv2.inRange(hsv, PURPLE_GLOVE_LOW, PURPLE_GLOVE_HIGH)
        # Keep the area where raised gloves appear in the supplied camera view.
        hand_y0, hand_y1 = int(h * 0.08), int(h * 0.96)
        mask[:hand_y0, :] = 0
        mask[hand_y1:, :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        now = time.monotonic()
        centers, punches = [None, None], []
        boxes = [None, None]
        tracked_mask = np.zeros((h, w), dtype=np.uint8)
        for side in (0, 1):
            x0, x1 = (0, w // 2) if side == 0 else (w // 2, w)
            y0 = hand_y0
            contours, _ = cv2.findContours(mask[y0:hand_y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = max(600, int(w * h * 0.003))
            max_area = int(w * h * 0.12)
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                bx, by, bw, bh = cv2.boundingRect(contour)
                aspect = bw / max(1, bh)
                if min_area <= area <= max_area and 0.32 <= aspect <= 2.8:
                    moments = cv2.moments(contour)
                    if moments["m00"] > 0:
                        point = (int(moments["m10"] / moments["m00"]) + x0,
                                 int(moments["m01"] / moments["m00"]) + y0)
                        candidates.append((contour, area, point, (x0 + bx, y0 + by, bw, bh)))
            if not candidates:
                self.previous[side] = None
                self.previous_time[side] = None
                self.smooth_centers[side] = None
                continue
            old_smooth = self.smooth_centers[side]
            if old_smooth is None:
                selected = max(candidates, key=lambda item: item[1])
            else:
                nearby = [item for item in candidates
                          if math.dist(old_smooth, item[2]) < w * 0.28]
                selected = min(nearby, key=lambda item: math.dist(old_smooth, item[2])) if nearby else max(candidates, key=lambda item: item[1])
            contour, _, point, boxes[side] = selected
            cv2.drawContours(tracked_mask, [contour], -1, 255, -1, offset=(x0, y0))
            old_smooth = self.smooth_centers[side]
            if old_smooth is None:
                smooth = point
            else:
                # Exponential smoothing reduces contour jitter while keeping punches responsive.
                smooth = (int(old_smooth[0] * 0.38 + point[0] * 0.62),
                          int(old_smooth[1] * 0.38 + point[1] * 0.62))
            self.smooth_centers[side] = smooth
            centers[side] = smooth
            prev = self.previous[side]
            prev_time = self.previous_time[side]
            elapsed = now - prev_time if prev_time is not None else 1.0
            if prev and elapsed < 0.20 and now - self.last_punch[side] > 0.62:
                dx, dy = point[0] - prev[0], point[1] - prev[1]
                # Only count a fast, deliberate inward thrust. Vertical sway and
                # small contour jitter are not punches.
                threshold = max(38, int(w * 0.055))
                toward_center = dx >= threshold if side == 0 else dx <= -threshold
                quick_jab = toward_center
                if quick_jab:
                    punches.append(side)
                    self.last_punch[side] = now
            self.previous[side] = point
            self.previous_time[side] = now

        hands_joined = False
        dodge = None
        if centers[0] and centers[1]:
            distance = math.dist(centers[0], centers[1])
            hands_joined = distance < w * 0.20 and abs(centers[0][1] - centers[1][1]) < h * 0.18
        # Two inward-moving gloves together are a join/dodge gesture, not a double jab.
        if len(punches) > 1:
            punches.clear()
        if hands_joined and not self.hands_were_joined and now - self.last_dodge > 0.55:
            dodge = (0.0, 1.0)  # joining the gloves performs a quick duck
            self.last_dodge = now
        self.hands_were_joined = hands_joined
        guarding = False

        # Equalize only the display value channel; glove segmentation uses raw HSV.
        preview_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        preview_hsv[:, :, 2] = cv2.equalizeHist(preview_hsv[:, :, 2])
        preview = cv2.cvtColor(preview_hsv, cv2.COLOR_HSV2BGR)
        green_tint = np.zeros_like(preview)
        green_tint[:] = (40, 230, 70)
        blended = cv2.addWeighted(preview, 0.55, green_tint, 0.45, 0)
        preview[tracked_mask > 0] = blended[tracked_mask > 0]
        cv2.putText(preview, "PURPLE GLOVE TRACK  L/R", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2)
        for i, point in enumerate(centers):
            if point:
                bx, by, bw, bh = boxes[i]
                cv2.rectangle(preview, (bx, by), (bx + bw, by + bh), (255, 170, 30), 2)
                cv2.circle(preview, point, 13, (80, 255, 120), 3)
                cv2.putText(preview, "L" if i == 0 else "R", (point[0] + 12, point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 120), 2)
        return punches, guarding, dodge, preview


def text(surface, value, size, color, pos, center=False, bold=False):
    # A bitmap-like monospace face keeps the UI consistent with the pixel art.
    font = pygame.font.SysFont("consolas", size, bold=bold)
    image = font.render(value, False, color)
    rect = image.get_rect()
    rect.center = pos if center else rect.center
    if not center:
        rect.topleft = pos
    surface.blit(image, rect)


def smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _camera_vector(value):
    """Normalize either a directional dodge tuple or the older scalar form."""
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return (max(-1.0, min(1.0, float(value[0]))),
                max(-1.0, min(1.0, float(value[1]))))
    try:
        return max(-1.0, min(1.0, float(value or 0.0))), 0.0
    except (TypeError, ValueError):
        return 0.0, 0.0


def load_game_sounds():
    """Load the generated arcade sound effects; keep the game usable without audio."""
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        pygame.mixer.set_num_channels(16)
    except pygame.error:
        return {}

    root = Path(__file__).resolve().parent / "assets" / "boxing" / "audio"
    sounds = {}
    for name in ("punch_swing", "hit", "player_hit", "guard", "block", "dodge", "enemy_windup", "round_bell"):
        path = root / f"{name}.wav"
        try:
            sound = pygame.mixer.Sound(str(path))
            sound.set_volume(0.45 if name not in ("round_bell", "enemy_windup") else 0.35)
            sounds[name] = sound
        except (pygame.error, OSError):
            continue
    return sounds


def play_game_sound(sounds, name):
    sound = sounds.get(name)
    if sound is not None:
        try:
            sound.play()
        except pygame.error:
            pass


def load_boxing_assets():
    """Load the extracted sprite pack once; keep procedural fallbacks available."""
    if not hasattr(load_boxing_assets, "cache"):
        root = Path(__file__).resolve().parent / "assets" / "boxing"
        files = {
            "ring": "ring_background.png",
            "enemy_stand": "enemy_pov_guard_new.png",
            "enemy_punch": "enemy_punch.png",
            "enemy_hit_left": "enemy_hit_left.png",
            "enemy_hit_right": "enemy_hit_right.png",
            "glove_left": "player_glove_idle_left.png",
            "glove_right": "player_glove_idle_right.png",
            "glove_punch_left": "player_punch_left.png",
            "glove_punch_right": "player_punch_right.png",
        }
        assets = {}
        for key, filename in files.items():
            path = root / filename
            if path.exists():
                try:
                    image = pygame.image.load(str(path))
                    assets[key] = image.convert() if key == "ring" else image.convert_alpha()
                except (pygame.error, OSError):
                    assets[key] = None
            else:
                assets[key] = None
        load_boxing_assets.cache = assets
    return load_boxing_assets.cache


def draw_glove(surface, x, y, scale, color=(190, 43, 42)):
    # Chunky stepped silhouette and hard-edged highlights keep the pixel-art feel.
    s = scale
    pygame.draw.polygon(surface, (42, 31, 32), [(x + 5*s, y + 22*s), (x + 17*s, y + 5*s),
                         (x + 55*s, y + 3*s), (x + 76*s, y + 18*s), (x + 81*s, y + 48*s),
                         (x + 66*s, y + 68*s), (x + 14*s, y + 66*s), (x - 3*s, y + 48*s)])
    pygame.draw.polygon(surface, color, [(x + 4*s, y + 19*s), (x + 17*s, y + 4*s),
                         (x + 54*s, y + 2*s), (x + 73*s, y + 16*s), (x + 77*s, y + 43*s),
                         (x + 63*s, y + 60*s), (x + 15*s, y + 60*s), (x, y + 43*s)])
    light = tuple(min(255, c + 48) for c in color)
    shadow = tuple(max(0, c - 46) for c in color)
    pygame.draw.rect(surface, light, (x + 17*s, y + 9*s, 34*s, 5*s))
    pygame.draw.rect(surface, light, (x + 8*s, y + 19*s, 5*s, 17*s))
    pygame.draw.polygon(surface, tuple(min(255, c + 24) for c in color),
                        [(x + 20*s, y + 27*s), (x + 48*s, y + 24*s),
                         (x + 59*s, y + 35*s), (x + 54*s, y + 43*s),
                         (x + 27*s, y + 42*s)])
    pygame.draw.line(surface, shadow, (x + 16*s, y + 44*s),
                     (x + 57*s, y + 50*s), max(1, int(2*s)))
    for stitch in (0, 1, 2):
        pygame.draw.rect(surface, light,
                         (x + (22 + stitch * 9)*s, y + 48*s, max(1, int(3*s)), max(1, int(2*s))))
    pygame.draw.rect(surface, shadow, (x + 62*s, y + 27*s, 8*s, 22*s))
    pygame.draw.rect(surface, (235, 226, 205), (x + 13*s, y + 49*s, 49*s, 16*s))
    pygame.draw.rect(surface, (171, 157, 137), (x + 13*s, y + 61*s, 49*s, 5*s))
    pygame.draw.rect(surface, (250, 244, 221), (x + 18*s, y + 52*s, 7*s, 3*s))
    pygame.draw.rect(surface, (121, 109, 104), (x + 31*s, y + 51*s, 3*s, 12*s))


def draw_opponent(surface, now, attack, hurt, dodge):
    """Dark boxing arena with blocky crowd, blue canvas and classic ring ropes."""
    assets = load_boxing_assets()
    if assets.get("ring") is not None:
        surface.fill((5, 7, 13))
        ring_size = (WIDTH, HEIGHT - 105)
        if not hasattr(draw_opponent, "ring_scaled"):
            draw_opponent.ring_scaled = pygame.transform.scale(assets["ring"], ring_size)
        surface.blit(draw_opponent.ring_scaled, (0, 105))
        return
    pygame.draw.rect(surface, (5, 7, 13), (0, 0, WIDTH, HEIGHT))
    pygame.draw.rect(surface, (13, 17, 29), (0, 205, WIDTH, 285))
    # Upper stand and distant lights.
    for i in range(9):
        x = 34 + i * 130
        pygame.draw.rect(surface, (25, 31, 45), (x, 116, 48, 5))
        pygame.draw.rect(surface, (146, 185, 222), (x + 17, 118, 14, 3))
        pygame.draw.polygon(surface, (20, 24, 37),
                            [(x + 14, 121), (x + 33, 121), (x + 110, 303), (x - 62, 303)])

    # Individual spectators create a busy tournament crowd in chunky pixel shapes.
    shirts = [(50, 65, 93), (101, 37, 48), (45, 91, 78), (114, 78, 50),
              (78, 58, 102), (42, 77, 112), (113, 107, 97)]
    for row in range(4):
        for i in range(31):
            x = i * 38 - 10 + (row % 2) * 18
            y = 239 + row * 47 + ((i * 13 + row * 7) % 10)
            shirt = shirts[(i * 5 + row * 3) % len(shirts)]
            skin = [(197, 133, 103), (222, 164, 132), (147, 95, 77), (193, 151, 112)][(i + row) % 4]
            hair = (32, 28, 31) if (i + row) % 3 else (85, 53, 40)
            pygame.draw.rect(surface, (7, 9, 15), (x - 3, y + 12, 28, 31))
            pygame.draw.rect(surface, shirt, (x, y + 13, 22, 27))
            pygame.draw.rect(surface, skin, (x + 4, y, 15, 16))
            pygame.draw.rect(surface, hair, (x + 3, y - 2, 17, 6))
            pygame.draw.rect(surface, (23, 20, 23), (x + 7, y + 7, 3, 2))
            pygame.draw.rect(surface, (23, 20, 23), (x + 14, y + 7, 3, 2))
            # A few raised arms and camera flashes animate the audience silhouette.
            if (i + row * 2) % 9 == 0:
                lift = int(5 + 3 * math.sin(now * 3 + i))
                pygame.draw.rect(surface, skin, (x - 5, y + 3 - lift, 5, 18 + lift))
            if (i + row) % 13 == 0:
                pygame.draw.rect(surface, (212, 231, 246), (x + 22, y + 2, 4, 7))

    # Ring floor with a strong perspective edge and subtle canvas seams.
    pygame.draw.polygon(surface, (99, 128, 150), [(0, 477), (WIDTH, 477), (WIDTH, HEIGHT), (0, HEIGHT)])
    pygame.draw.polygon(surface, (136, 162, 179), [(0, 492), (WIDTH, 492), (WIDTH, HEIGHT), (0, HEIGHT)])
    pygame.draw.line(surface, (186, 202, 209), (0, 493), (WIDTH, 493), 5)
    for x in range(-420, WIDTH + 500, 175):
        pygame.draw.line(surface, (105, 131, 150), (WIDTH // 2, 496), (x, HEIGHT), 2)
    # Turnbuckle posts frame the first-person view.
    for x, color in ((54, (57, 104, 211)), (WIDTH - 92, (61, 111, 223))):
        pygame.draw.rect(surface, (3, 5, 11), (x - 7, 267, 40, 257))
        pygame.draw.rect(surface, color, (x, 270, 25, 245))
        pygame.draw.rect(surface, (125, 166, 250), (x + 4, 277, 7, 224))
        pygame.draw.rect(surface, (21, 40, 83), (x + 18, 277, 7, 238))
    # Red, white and blue ring ropes sit behind both boxers.
    ropes = ((331, (220, 48, 49)), (379, (202, 216, 224)),
             (427, (55, 106, 219)), (475, (205, 221, 229)))
    for y, color in ropes:
        pygame.draw.line(surface, (2, 4, 9), (0, y + 5), (WIDTH, y + 5), 9)
        pygame.draw.line(surface, color, (0, y), (WIDTH, y), 5)
        pygame.draw.line(surface, tuple(min(255, channel + 28) for channel in color),
                         (0, y - 1), (WIDTH, y - 1), 1)

def draw_boxer(surface, now, attack, hurt, dodge):
    """Detailed front-facing pixel boxer, rendered separately for close POV framing."""
    cx = WIDTH // 2
    bob = 5 if hurt else 0
    # The rival stays planted in the ring; only the camera and scenery shift on a dodge.
    offset = int(math.sin(now * 38) * 7) if hurt else 0
    # legs, shorts, torso, head
    # Shorts, knees and lower legs
    pygame.draw.polygon(surface, (24, 43, 126), [(cx - 72 + offset, 483 + bob), (cx + 72 + offset, 483 + bob),
                         (cx + 62 + offset, 552), (cx + 8 + offset, 559), (cx - 8 + offset, 559), (cx - 65 + offset, 552)])
    pygame.draw.rect(surface, (241, 239, 228), (cx - 68 + offset, 491 + bob, 132, 12))
    pygame.draw.rect(surface, (54, 99, 224), (cx - 9 + offset, 492 + bob, 18, 10))
    pygame.draw.polygon(surface, (161, 91, 72), [(cx - 58 + offset, 553), (cx - 9 + offset, 557),
                         (cx - 18 + offset, 637), (cx - 53 + offset, 637)])
    pygame.draw.polygon(surface, (195, 119, 91), [(cx + 9 + offset, 557), (cx + 57 + offset, 553),
                         (cx + 52 + offset, 637), (cx + 18 + offset, 637)])
    pygame.draw.rect(surface, (220, 213, 194), (cx - 53 + offset, 615, 36, 9))
    pygame.draw.rect(surface, (220, 213, 194), (cx + 18 + offset, 615, 36, 9))
    pygame.draw.rect(surface, (40, 43, 51), (cx - 58 + offset, 632, 47, 21))
    pygame.draw.rect(surface, (40, 43, 51), (cx + 12 + offset, 632, 47, 21))
    # Bare upper body with broad shoulders and layered muscle shading.
    pygame.draw.polygon(surface, (111, 57, 48), [(cx - 83 + offset, 248 + bob), (cx - 50 + offset, 218 + bob),
                         (cx + 50 + offset, 218 + bob), (cx + 83 + offset, 248 + bob),
                         (cx + 88 + offset, 405 + bob), (cx + 73 + offset, 462 + bob),
                         (cx - 73 + offset, 462 + bob), (cx - 88 + offset, 405 + bob)])
    pygame.draw.polygon(surface, (224, 145, 111), [(cx - 76 + offset, 245 + bob), (cx - 46 + offset, 223 + bob),
                         (cx + 46 + offset, 223 + bob), (cx + 76 + offset, 245 + bob),
                         (cx + 78 + offset, 402 + bob), (cx + 66 + offset, 452 + bob),
                         (cx - 66 + offset, 452 + bob), (cx - 78 + offset, 402 + bob)])
    pygame.draw.polygon(surface, (247, 175, 137), [(cx - 70 + offset, 270 + bob), (cx - 20 + offset, 260 + bob),
                         (cx - 8 + offset, 328 + bob), (cx - 81 + offset, 312 + bob)])
    pygame.draw.polygon(surface, (247, 175, 137), [(cx + 70 + offset, 270 + bob), (cx + 20 + offset, 260 + bob),
                         (cx + 8 + offset, 328 + bob), (cx + 81 + offset, 312 + bob)])
    pygame.draw.polygon(surface, (157, 84, 69), [(cx - 4 + offset, 316 + bob),
                         (cx - 66 + offset, 302 + bob), (cx - 50 + offset, 344 + bob),
                         (cx - 5 + offset, 351 + bob)])
    pygame.draw.polygon(surface, (157, 84, 69), [(cx + 4 + offset, 316 + bob),
                         (cx + 66 + offset, 302 + bob), (cx + 50 + offset, 344 + bob),
                         (cx + 5 + offset, 351 + bob)])
    pygame.draw.line(surface, (255, 194, 150), (cx - 45 + offset, 319 + bob),
                     (cx - 8 + offset, 329 + bob), 3)
    pygame.draw.line(surface, (255, 194, 150), (cx + 45 + offset, 319 + bob),
                     (cx + 8 + offset, 329 + bob), 3)
    pygame.draw.rect(surface, (155, 106, 82), (cx - 5 + offset, 326 + bob, 10, 18))
    pygame.draw.rect(surface, (164, 111, 84), (cx - 4 + offset, 357 + bob, 8, 8))
    for ab_y in (372, 397, 422):
        for ab_x in (-18, 5):
            pygame.draw.rect(surface, (145, 73, 61),
                             (cx + ab_x + offset, ab_y + bob, 13, 14))
            pygame.draw.rect(surface, (239, 161, 123),
                             (cx + ab_x + 2 + offset, ab_y + 2 + bob, 9, 4))
    pygame.draw.line(surface, (118, 62, 53), (cx, 370 + bob), (cx, 436 + bob), 3)
    for px, py, color in ((-62, 283, (231, 172, 129)), (-47, 291, (146, 91, 74)),
                          (52, 276, (232, 174, 130)), (67, 293, (146, 91, 74)),
                          (-52, 350, (229, 168, 126)), (43, 352, (143, 91, 75)),
                          (-58, 385, (218, 157, 118)), (55, 414, (145, 91, 75)),
                          (-38, 430, (224, 163, 121)), (29, 440, (145, 91, 75))):
        pygame.draw.rect(surface, color, (cx + px + offset, py + bob, 5, 4))
    pygame.draw.polygon(surface, (111, 57, 48), [(cx - 24 + offset, 174 + bob), (cx + 24 + offset, 174 + bob),
                         (cx + 26 + offset, 220 + bob), (cx - 26 + offset, 220 + bob)])
    pygame.draw.polygon(surface, (224, 145, 111), [(cx - 20 + offset, 177 + bob), (cx + 20 + offset, 177 + bob),
                         (cx + 22 + offset, 219 + bob), (cx - 22 + offset, 219 + bob)])
    pygame.draw.rect(surface, (244, 169, 127), (cx - 14 + offset, 186 + bob, 6, 28))
    pygame.draw.rect(surface, (151, 103, 81), (cx + 13 + offset, 191 + bob, 5, 27))
    pygame.draw.rect(surface, (37, 30, 28), (cx - 51 + offset, 119 + bob, 102, 21))
    # Face and hair pixels make the opponent readable at a glance.
    pygame.draw.rect(surface, (116, 75, 61), (cx - 50 + offset, 109 + bob, 100, 70))
    pygame.draw.rect(surface, (192, 139, 103), (cx - 43 + offset, 102 + bob, 86, 76))
    pygame.draw.rect(surface, (214, 160, 119), (cx - 36 + offset, 110 + bob, 13, 47))
    pygame.draw.rect(surface, (172, 119, 91), (cx + 30 + offset, 117 + bob, 9, 45))
    pygame.draw.rect(surface, (49, 37, 34), (cx - 46 + offset, 98 + bob, 92, 20))
    pygame.draw.rect(surface, (49, 37, 34), (cx - 46 + offset, 110 + bob, 13, 25))
    pygame.draw.rect(surface, (49, 37, 34), (cx + 33 + offset, 110 + bob, 13, 25))
    # Pixel face and short hair; keep the opponent's eyes visible for close POV.
    pygame.draw.rect(surface, (49, 31, 38), (cx - 46 + offset, 98 + bob, 92, 20))
    pygame.draw.rect(surface, (49, 31, 38), (cx - 46 + offset, 110 + bob, 13, 25))
    pygame.draw.rect(surface, (49, 31, 38), (cx + 33 + offset, 110 + bob, 13, 25))
    pygame.draw.polygon(surface, (67, 39, 44), [(cx - 31 + offset, 129 + bob),
                         (cx - 9 + offset, 125 + bob), (cx - 7 + offset, 132 + bob),
                         (cx - 22 + offset, 134 + bob)])
    pygame.draw.polygon(surface, (67, 39, 44), [(cx + 31 + offset, 129 + bob),
                         (cx + 9 + offset, 125 + bob), (cx + 7 + offset, 132 + bob),
                         (cx + 22 + offset, 134 + bob)])
    pygame.draw.rect(surface, (35, 29, 35), (cx - 25 + offset, 136 + bob, 13, 6))
    pygame.draw.rect(surface, (35, 29, 35), (cx + 12 + offset, 136 + bob, 13, 6))
    pygame.draw.rect(surface, (244, 206, 157), (cx - 22 + offset, 137 + bob, 4, 4))
    pygame.draw.rect(surface, (244, 206, 157), (cx + 15 + offset, 137 + bob, 4, 4))
    pygame.draw.rect(surface, (151, 99, 78), (cx - 5 + offset, 142 + bob, 10, 20))
    pygame.draw.rect(surface, (219, 156, 115), (cx + 3 + offset, 146 + bob, 6, 8))
    # Short pixel beard and jaw shadow add definition to the close-up face.
    pygame.draw.polygon(surface, (54, 36, 39), [(cx - 31 + offset, 151 + bob),
                         (cx - 23 + offset, 158 + bob), (cx - 14 + offset, 161 + bob),
                         (cx - 7 + offset, 158 + bob), (cx + 7 + offset, 158 + bob),
                         (cx + 14 + offset, 161 + bob), (cx + 23 + offset, 158 + bob),
                         (cx + 31 + offset, 151 + bob), (cx + 26 + offset, 172 + bob),
                         (cx + 15 + offset, 179 + bob), (cx - 15 + offset, 179 + bob),
                         (cx - 26 + offset, 172 + bob)])
    pygame.draw.rect(surface, (87, 53, 48), (cx - 34 + offset, 111 + bob, 7, 22))
    pygame.draw.rect(surface, (230, 168, 122), (cx - 31 + offset, 143 + bob, 5, 7))
    pygame.draw.rect(surface, (230, 168, 122), (cx + 27 + offset, 143 + bob, 5, 7))
    pygame.draw.rect(surface, (43, 29, 34), (cx - 14 + offset, 168 + bob, 28, 4))
    pygame.draw.rect(surface, (224, 173, 132), (cx - 7 + offset, 175 + bob, 14, 2))
    # Arms visibly pull back during the warning, then extend smoothly toward camera.
    glove_y = 169 + bob
    side, extension, charge = attack if attack else (None, 0.0, 0.0)
    left_x = cx - 148 + offset - int(28 * charge) + int(75 * extension if side == "left" else 0)
    right_x = cx + 68 + offset + int(28 * charge) - int(75 * extension if side == "right" else 0)
    pygame.draw.polygon(surface, (111, 57, 48), [
        (cx - 72 + offset, 252 + bob), (cx - 128 + offset, 201 + bob),
        (cx - 162 + offset, 207 + bob), (cx - 178 + offset, 237 + bob),
        (cx - 149 + offset, 267 + bob), (cx - 105 + offset, 267 + bob)])
    pygame.draw.polygon(surface, (224, 145, 111), [
        (cx - 74 + offset, 251 + bob), (cx - 127 + offset, 208 + bob),
        (cx - 157 + offset, 214 + bob), (cx - 169 + offset, 237 + bob),
        (cx - 145 + offset, 258 + bob), (cx - 106 + offset, 260 + bob)])
    pygame.draw.polygon(surface, (111, 57, 48), [
        (cx + 72 + offset, 252 + bob), (cx + 128 + offset, 201 + bob),
        (cx + 162 + offset, 207 + bob), (cx + 178 + offset, 237 + bob),
        (cx + 149 + offset, 267 + bob), (cx + 105 + offset, 267 + bob)])
    pygame.draw.polygon(surface, (224, 145, 111), [
        (cx + 74 + offset, 251 + bob), (cx + 127 + offset, 208 + bob),
        (cx + 157 + offset, 214 + bob), (cx + 169 + offset, 237 + bob),
        (cx + 145 + offset, 258 + bob), (cx + 106 + offset, 260 + bob)])
    glove_color = (226, 27, 43) if charge > 0.15 else (188, 20, 35)
    glove_scale = 1.42 + 0.18 * extension
    draw_glove(surface, left_x, glove_y, glove_scale, glove_color)
    draw_glove(surface, right_x, glove_y, glove_scale, glove_color)
    if hurt:
        spark = (255, 225, 116)
        for sx, sy in ((cx - 71, 111), (cx + 67, 125), (cx + 54, 91)):
            pygame.draw.line(surface, spark, (sx - 7, sy), (sx + 7, sy), 3)
            pygame.draw.line(surface, spark, (sx, sy - 7), (sx, sy + 7), 3)

def draw_arena(screen, now, player_hp, rival_hp, combo, status, attack, hurt, dodge, guard,
               player_punch=None, player_ability=None, enemy_ability=None,
               counter_open=False, rival_hurt=False, round_started=None, rival_hit_side=0):
    camera_x, camera_y = _camera_vector(dodge)
    screen.fill((28, 31, 38))
    pygame.draw.polygon(screen, (90, 96, 96), [(0, 0), (WIDTH, 0), (WIDTH, 335), (0, 335)])
    # Keep the arena as a distant plane; only this layer pans with the head dodge.
    opponent_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    draw_opponent(opponent_layer, now, attack, hurt, 0.0)
    zoom = 1.05
    enlarged = pygame.transform.scale(opponent_layer, (int(WIDTH * zoom), int(HEIGHT * zoom)))
    crop_x = (enlarged.get_width() - WIDTH) // 2 + int(camera_x * 16)
    crop_y = max(0, int(HEIGHT * (zoom - 1.0) * 0.08 + camera_y * 8))
    bg_crop = enlarged.subsurface(pygame.Rect(crop_x, crop_y, WIDTH, HEIGHT))
    screen.blit(bg_crop, (0, 0))
    # Render the opponent separately so he remains anchored while the arena moves.
    # The close-up POV cutout stays centered. A small forward scale pulse sells
    # the opponent's lunge without replacing this user-selected character art.
    assets = load_boxing_assets()
    enemy_stand = assets.get("enemy_stand")
    punch_extension = attack[1] if attack else 0.0
    hit_key = "enemy_hit_left" if rival_hit_side == 0 else "enemy_hit_right"
    if rival_hurt and assets.get(hit_key) is not None:
        fighter_sprite = assets[hit_key]
    elif punch_extension > 0.16 and assets.get("enemy_punch") is not None:
        fighter_sprite = assets["enemy_punch"]
    else:
        fighter_sprite = enemy_stand
    if fighter_sprite is not None:
        fighter_height = 700 + int(55 * punch_extension)
        fighter_width = max(1, int(fighter_sprite.get_width() * fighter_height / fighter_sprite.get_height()))
        fighter_scaled = pygame.transform.scale(fighter_sprite, (fighter_width, fighter_height))
        impact_shake = int(math.sin(now * 42) * 6) if rival_hurt else 0
        screen.blit(fighter_scaled, (WIDTH // 2 - fighter_width // 2 + impact_shake, 100))
    else:
        fighter = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        draw_boxer(fighter, now, attack, rival_hurt, 0.0)
        fighter_zoom = 1.38
        fighter_scaled = pygame.transform.scale(
            fighter, (int(WIDTH * fighter_zoom), int(HEIGHT * fighter_zoom)))
        fighter_crop_x = (fighter_scaled.get_width() - WIDTH) // 2
        fighter_crop_y = int(HEIGHT * (fighter_zoom - 1.0) * 0.08)
        screen.blit(fighter_scaled.subsurface(
            pygame.Rect(fighter_crop_x, fighter_crop_y, WIDTH, HEIGHT)), (0, 0))
    # First-person sleeves and the player's gloves
    bob = int(math.sin(now * 5) * 4)
    glove_raise = 16 * guard
    left_lunge = player_punch[1] * 115 if player_punch and player_punch[0] == 0 else 0
    right_lunge = player_punch[1] * 115 if player_punch and player_punch[0] == 1 else 0
    hand_sway_x, hand_sway_y = int(camera_x * 12), int(camera_y * -9)
    if assets.get("glove_left") is not None and assets.get("glove_right") is not None:
        glove_height = 390
        left_width = int(assets["glove_left"].get_width() * glove_height / assets["glove_left"].get_height())
        right_width = int(assets["glove_right"].get_width() * glove_height / assets["glove_right"].get_height())
        left_glove = pygame.transform.scale(assets["glove_left"], (left_width, glove_height))
        right_glove = pygame.transform.scale(assets["glove_right"], (right_width, glove_height))
        left_x = -12 + int(left_lunge) + hand_sway_x
        right_x = WIDTH - right_width + 12 - int(right_lunge) + hand_sway_x
        glove_y = HEIGHT - glove_height + 15 - int(glove_raise) + hand_sway_y
        punch_strength = max(0.0, min(1.0, player_punch[1])) if player_punch else 0.0
        if not player_punch or player_punch[0] != 0:
            screen.blit(left_glove, (left_x, glove_y - int(left_lunge * 0.12)))
        if not player_punch or player_punch[0] != 1:
            screen.blit(right_glove, (right_x, glove_y - int(right_lunge * 0.12)))
        if player_punch and punch_strength > 0.0:
            punch_key = "glove_punch_left" if player_punch[0] == 0 else "glove_punch_right"
            punch_sprite = assets.get(punch_key)
            if punch_sprite is not None:
                punch_height = 650
                punch_width = int(punch_sprite.get_width() * punch_height / punch_sprite.get_height())
                punch_sprite = pygame.transform.scale(punch_sprite, (punch_width, punch_height))
                inward = int(95 * punch_strength) * (-1 if player_punch[0] == 0 else 1)
                punch_x = (WIDTH - punch_width) // 2 + inward + hand_sway_x
                punch_y = HEIGHT - punch_height + 150 + hand_sway_y
                screen.blit(punch_sprite, (punch_x, punch_y))
    else:
        pygame.draw.polygon(screen, (39, 52, 75), [(0, HEIGHT), (0, 570 + bob), (118, 558 + bob), (205, HEIGHT)])
        pygame.draw.polygon(screen, (39, 52, 75), [(WIDTH, HEIGHT), (WIDTH, 570 - bob), (WIDTH - 118, 558 - bob), (WIDTH - 205, HEIGHT)])
        draw_glove(screen, -24 + left_lunge + hand_sway_x, 496 - glove_raise + bob - left_lunge * 0.18 + hand_sway_y, 3.0, (160, 79, 81))
        draw_glove(screen, WIDTH - 238 - right_lunge + hand_sway_x, 496 - glove_raise - bob - right_lunge * 0.18 + hand_sway_y, 3.0, (160, 79, 81))

    # HUD
    hud = pygame.Surface((WIDTH, 105), pygame.SRCALPHA)
    hud.fill((3, 4, 8, 245))
    screen.blit(hud, (0, 0))
    text(screen, "PLAYER", 21, (244, 244, 239), (30, 5), bold=True)
    text(screen, "OPPONENT", 21, (244, 244, 239), (870, 5), center=True, bold=True)
    # White pixel frames and red health blocks echo the reference HUD.
    for x in (25, WIDTH - 425):
        pygame.draw.rect(screen, (238, 239, 235), (x, 31, 400, 28), 3)
        pygame.draw.rect(screen, (12, 13, 17), (x + 6, 37, 388, 16))
    pygame.draw.rect(screen, (232, 41, 48), (31, 39, int(376 * player_hp / 100), 12))
    opponent_width = int(376 * rival_hp / 100)
    pygame.draw.rect(screen, (232, 41, 48),
                     (WIDTH - 31 - opponent_width, 39, opponent_width, 12))
    text(screen, "ROUND 1", 16, (93, 211, 195), (WIDTH // 2, 29), center=True, bold=True)
    seconds_left = 150 if round_started is None else max(0, int(150 - (now - round_started)))
    text(screen, f"{seconds_left // 60}:{seconds_left % 60:02d}", 19, (244, 244, 239),
         (WIDTH // 2, 52), center=True, bold=True)
    if player_ability:
        remaining = max(0.0, player_ability[1] - now)
        pygame.draw.rect(screen, (76, 46, 20), (28, 69, 160, 15))
        pygame.draw.rect(screen, (255, 160, 55), (28, 69, int(160 * remaining / 8), 15))
        text(screen, f"ADRENALIN {remaining:.1f}s", 11, (255, 255, 255), (34, 69), bold=True)
    if enemy_ability:
        remaining = max(0.0, enemy_ability[1] - now)
        x = WIDTH - 240
        pygame.draw.rect(screen, (70, 28, 32), (x, 69, 160, 15))
        pygame.draw.rect(screen, (235, 70, 65), (x, 69, int(160 * remaining / 8), 15))
        text(screen, f"BERSERK {remaining:.1f}s", 11, (255, 255, 255), (x + 6, 69), bold=True)
    text(screen, f"COMBO {combo}", 13, (255, 221, 132), (WIDTH // 2, 83), center=True, bold=True)
    if status:
        text(screen, status, 20, (255, 236, 173), (25, 118), bold=True)
    if attack and attack[2] > 0.1:
        side_label = "SERANGAN KIRI  ←" if attack[0] == "left" else "→  SERANGAN KANAN"
        warn = (255, 117, 91) if attack[2] > 0.65 else (255, 213, 126)
        text(screen, side_label, 16, warn, (WIDTH // 2, 151), center=True, bold=True)
        pygame.draw.rect(screen, (65, 39, 36), (25, 148, 210, 12))
        pygame.draw.rect(screen, (247, 150, 71), (25, 148, int(210 * attack[2]), 12))
    if counter_open:
        # Brief arcade-style opening cue after a clean dodge or block.
        pulse = 185 + int(55 * (0.5 + 0.5 * math.sin(now * 12)))
        pygame.draw.rect(screen, (22, 45, 43), (WIDTH // 2 - 116, 112, 232, 30), border_radius=5)
        pygame.draw.rect(screen, (pulse, 225, 151), (WIDTH // 2 - 116, 112, 232, 30), 2, border_radius=5)
        text(screen, "COUNTER! LAWAN TERBUKA", 14, (pulse, 255, 184),
             (WIDTH // 2, 119), center=True, bold=True)

    # Subtle pixel vignette frames the action like a retro sports game.
    vignette = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for i in range(6):
        alpha = 10 + i * 3
        inset = i * 9
        pygame.draw.rect(vignette, (9, 11, 18, alpha), (inset, inset, WIDTH - inset * 2, HEIGHT - inset * 2), 9)
    screen.blit(vignette, (0, 0))
    if hurt:
        hit_flash = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        hit_flash.fill((190, 24, 32, 34))
        screen.blit(hit_flash, (0, 0))
def main():
    pygame.init()
    sounds = load_game_sounds()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Nyna Boxing — First Person")
    clock = pygame.time.Clock()
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    camera_ready = camera.isOpened()
    tracker = MotionTracker()
    mode = "ready"
    player_hp = rival_hp = 100.0
    player_ability_used = enemy_ability_used = False
    player_ability_until = enemy_ability_until = 0.0
    next_attack = time.monotonic() + 3
    attack_start = hurt_until = action_until = 0.0
    attack_side = None
    attack_resolved = False
    counter_window_until = 0.0
    rival_hurt_until = 0.0
    dodge_until = 0.0
    dodge_pose = guard_pose = 0.0
    dodge_direction_x = dodge_direction_y = 0.0
    camera_sway_x = camera_sway_y = 0.0
    player_punch_side, player_punch_start = 0, -10.0
    rival_hit_side = 0
    last_action = 0.0
    combo = 0
    status = ""
    round_started = None
    guard_was_active = False
    preview_surface = None
    running = True
    try:
        while running:
            dt = clock.tick(FPS) / 1000
            now = time.monotonic()
            punches, guarding, dodge = [], False, None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and mode in ("ready", "lost", "won", "draw"):
                        mode = "fight"
                        play_game_sound(sounds, "round_bell")
                        round_started = now
                        player_hp = rival_hp = 100.0
                        player_ability_used = enemy_ability_used = False
                        player_ability_until = enemy_ability_until = 0.0
                        next_attack = now + 3
                        attack_start = 0.0
                        attack_resolved = False
                        counter_window_until = 0.0
                        rival_hurt_until = 0.0
                        rival_hit_side = 0
                        combo = 0
                        guard_was_active = False
                    elif mode == "fight" and not camera_ready:
                        if event.key in (pygame.K_j, pygame.K_f):
                            punches.append(0)
                        elif event.key in (pygame.K_l, pygame.K_h):
                            punches.append(1)
                        elif event.key in (pygame.K_a, pygame.K_d):
                            dodge = (-1.0 if event.key == pygame.K_a else 1.0, 0.0)
                        elif event.key == pygame.K_b:
                            guarding = True

            ok, frame = camera.read() if camera_ready else (False, None)
            if ok:
                frame = cv2.flip(frame, 1)
                punches, guarding, dodge, preview = tracker.process(frame)
                preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                preview_surface = pygame.surfarray.make_surface(np.transpose(preview, (1, 0, 2)))

            if mode == "fight":
                if not camera_ready:
                    keys = pygame.key.get_pressed()
                    guarding = keys[pygame.K_b]
                    if keys[pygame.K_a] or keys[pygame.K_d]:
                        dodge = (-1.0 if keys[pygame.K_a] else 1.0, 0.0)
                else:
                    # Keep a keyboard block available while webcam controls are active.
                    guarding = guarding or pygame.key.get_pressed()[pygame.K_b]
                if dodge is not None:
                    dodge_direction_x, dodge_direction_y = _camera_vector(dodge)
                    dodge_until = now + 0.38
                    play_game_sound(sounds, "dodge")
                if guarding and not guard_was_active:
                    play_game_sound(sounds, "guard")
                guard_was_active = guarding
                guard_target = 1.0 if guarding else 0.0
                dodge_target = 1.0 if now < dodge_until else 0.0
                blend = 1.0 - math.exp(-dt * 11.0)
                guard_pose += (guard_target - guard_pose) * blend
                dodge_pose += (dodge_target - dodge_pose) * blend
                active_dodge = now < dodge_until
                sway_x_target = dodge_direction_x * dodge_pose if active_dodge else 0.0
                sway_y_target = dodge_direction_y * dodge_pose if active_dodge else 0.0
                camera_sway_x += (sway_x_target - camera_sway_x) * blend
                camera_sway_y += (sway_y_target - camera_sway_y) * blend

                attack_age = now - attack_start
                attack_duration = 1.18
                if now >= next_attack and (attack_start == 0.0 or attack_age >= attack_duration):
                    attack_side = random.choice(("left", "right"))
                    attack_start = now
                    attack_age = 0.0
                    attack_resolved = False
                    next_attack = now + (random.uniform(2.8, 3.5) if now < enemy_ability_until
                                         else random.uniform(4.0, 5.0))
                    status = "LAWAN MENGANCANG PUKULAN!"
                    play_game_sound(sounds, "enemy_windup")

                if attack_start:
                    attack_age = now - attack_start
                charge = 0.0
                extension = 0.0
                if 0.0 <= attack_age < 0.62:
                    charge = smoothstep(attack_age / 0.62)
                elif 0.62 <= attack_age < 0.85:
                    punch_phase = smoothstep((attack_age - 0.62) / 0.23)
                    charge = 1.0 - punch_phase
                    extension = punch_phase
                elif 0.85 <= attack_age < 1.03:
                    extension = 1.0 - smoothstep((attack_age - 0.85) / 0.18)

                if 0.76 <= attack_age < 0.92 and not attack_resolved:
                    if guard_pose > 0.45:
                        status, action_until = "TANGKIS BERHASIL!", now + 0.6
                        counter_window_until = now + 0.75
                        play_game_sound(sounds, "block")
                    elif dodge_pose > 0.45:
                        status, action_until = "HINDAR BERSIH! BALAS SEKARANG!", now + 0.6
                        counter_window_until = now + 1.25
                    else:
                        damage = 18 if now < enemy_ability_until else 14
                        player_hp = max(0, player_hp - damage)
                        status, hurt_until = "KENA PUKULAN!", now + 0.35
                        play_game_sound(sounds, "player_hit")
                    attack_resolved = True

                if punches and now >= last_action + 0.25:
                    play_game_sound(sounds, "punch_swing")
                    if now < counter_window_until:
                        punch_damage = 13 if now < player_ability_until else 9
                        rival_hp = max(0, rival_hp - punch_damage * len(punches))
                        combo += len(punches)
                        status = "COUNTER HIT!" if len(punches) == 1 else "KOMBINASI COUNTER!"
                        rival_hurt_until = now + 0.34
                        rival_hit_side = punches[-1]
                        play_game_sound(sounds, "hit")
                        # A clean counter staggers the rival and buys a readable beat.
                        next_attack = max(next_attack, now + 1.0)
                        counter_window_until = now + 0.45
                    else:
                        status = "PUNCH DITANGKIS — HINDARI SERANGAN DULU!"
                    action_until = now + 0.35
                    last_action = now
                    player_punch_side = punches[-1]
                    player_punch_start = now
                if player_hp <= 50 and not player_ability_used:
                    player_ability_used = True
                    player_ability_until = now + 8.0
                    action_until = now + 2.0
                    status = "ABILITY AKTIF: ADRENALIN! Pukulan lebih kuat 8 detik"
                if rival_hp <= 50 and not enemy_ability_used:
                    enemy_ability_used = True
                    enemy_ability_until = now + 8.0
                    next_attack = min(next_attack, now + 1.5)
                    action_until = now + 2.0
                    status = "ABILITY LAWAN: BERSERK! Serangan lebih cepat dan kuat"
                if now > action_until and (attack_age > 1.03 or attack_start == 0.0):
                    status = ""
                if player_hp <= 0:
                    mode = "lost"
                elif rival_hp <= 0:
                    mode = "won"
                elif round_started is not None and now - round_started >= 150:
                    mode = "won" if player_hp > rival_hp else "lost" if rival_hp > player_hp else "draw"
                    status = "WAKTU HABIS!"

            attack_pose = (attack_side, extension, charge) if mode == "fight" and attack_start and attack_age < 1.03 else None
            punch_age = now - player_punch_start
            punch_strength = 1.0 - abs(punch_age / 0.28 * 2 - 1) if 0 <= punch_age < 0.28 else 0.0
            player_punch = (player_punch_side, smoothstep(punch_strength)) if punch_strength > 0 else None
            draw_arena(screen, now, player_hp, rival_hp, combo, status, attack_pose,
                       now < hurt_until, (camera_sway_x, camera_sway_y), guard_pose, player_punch,
                       ("ADRENALIN", player_ability_until) if now < player_ability_until else None,
                       ("BERSERK", enemy_ability_until) if now < enemy_ability_until else None,
                       now < counter_window_until, now < rival_hurt_until, round_started,
                       rival_hit_side)
            if preview_surface:
                thumb = pygame.transform.smoothscale(preview_surface, (180, 135))
                screen.blit(thumb, (WIDTH - 195, 115))
                pygame.draw.rect(screen, (240, 240, 240), (WIDTH - 195, 115, 180, 135), 2)

            if mode != "fight":
                veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                veil.fill((8, 10, 16, 170))
                screen.blit(veil, (0, 0))
                if mode == "ready":
                    title = "NYNA ARCADE BOXING"
                    subtitle = "Baca ancang-ancang lawan, hindari pukulannya, lalu counter!"
                    controls = ("Ayunkan glove kiri/kanan untuk jab  •  Rapatkan kedua glove untuk menghindar  •  Tahan B untuk tangkis"
                                if camera_ready else "J/F atau L/H = pukul  •  A/D = hindar  •  B = tangkis  •  Pukul saat COUNTER terbuka")
                    prompt = "SPACE mulai  •  ESC keluar"
                elif mode == "lost":
                    title, subtitle, controls, prompt = "KNOCKOUT", "Kamu kalah dalam pertandingan ini.", "Coba kombinasi pukulan dan tangkisan.", "SPACE tanding lagi  •  ESC keluar"
                elif mode == "draw":
                    title, subtitle, controls, prompt = "SERI", "Waktu habis dan HP kedua petinju sama.", "Pertandingan berakhir imbang.", "SPACE tanding lagi  •  ESC keluar"
                else:
                    title, subtitle, controls, prompt = "KAMU JUARA!", "Lawan kalah atau HP-mu unggul saat waktu habis.", "Pertandingan selesai.", "SPACE tanding lagi  •  ESC keluar"
                text(screen, title, 42, (255, 255, 255), (WIDTH // 2, 260), center=True, bold=True)
                text(screen, subtitle, 21, (225, 228, 234), (WIDTH // 2, 315), center=True)
                text(screen, controls, 16, (255, 214, 141), (WIDTH // 2, 365), center=True)
                text(screen, prompt, 18, (126, 236, 165), (WIDTH // 2, 420), center=True, bold=True)
            pygame.display.flip()
    finally:
        camera.release()
        pygame.quit()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

