"""
Generates synthetic fundus sample images for the 5 ICDR Diabetic Retinopathy grades.
Creates anatomically realistic retinal fundus graphics with optic disc, macula,
vascular tree, and stage-specific retinal lesions for immediate testing and demonstration.
"""

import os
import cv2
import numpy as np


def generate_base_retina(size: int = 512) -> np.ndarray:
    """
    Creates an anatomically structured base retinal fundus image:
    - Dark circular outer border
    - Warm orange-red retinal pigment epithelium gradient
    - Optic disc (yellow-pink oval nasal to macula)
    - Macula and fovea (avascular zone, darker orange-brown)
    - Major retinal vessel arcades (superior and inferior temporal vessels)
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.44)
    
    # 1. Circular fundus background with radial gradient
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist_from_center <= radius
    
    # Radial color shading: central fundus is vibrant reddish-orange, periphery is slightly darker
    norm_dist = np.clip(dist_from_center / radius, 0, 1)
    
    red_channel = (190 - 45 * norm_dist).astype(np.uint8)
    green_channel = (85 - 35 * norm_dist).astype(np.uint8)
    blue_channel = (30 - 15 * norm_dist).astype(np.uint8)
    
    fundus = np.dstack([red_channel, green_channel, blue_channel])
    # Add subtle organic texture
    noise = np.random.normal(0, 4, (size, size, 3)).astype(np.int16)
    fundus_textured = np.clip(fundus.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    img[mask] = fundus_textured[mask]
    
    # 2. Optic Disc (yellow-pink oval on nasal side, say x = 360, y = 256 for left-eye representation)
    disc_center = (int(size * 0.70), int(size * 0.50))
    cv2.ellipse(img, disc_center, (int(size * 0.075), int(size * 0.09)), 0, 0, 360, (255, 220, 160), -1)
    # Optic cup (paler center)
    cv2.ellipse(img, disc_center, (int(size * 0.035), int(size * 0.045)), 0, 0, 360, (255, 245, 210), -1)
    
    # 3. Macula (darker avascular spot, x = 200, y = 256)
    macula_center = (int(size * 0.40), int(size * 0.50))
    cv2.circle(img, macula_center, int(size * 0.06), (145, 55, 25), -1)
    cv2.circle(img, macula_center, int(size * 0.02), (120, 40, 20), -1)  # Fovea
    
    # 4. Retinal Blood Vessel Arborescence (branching out from optic disc)
    vessel_color = (130, 25, 15)  # Dark burgundy blood vessel
    
    # Superior temporal arcade
    pts_sup = np.array([
        disc_center,
        (int(size * 0.60), int(size * 0.32)),
        (int(size * 0.42), int(size * 0.28)),
        (int(size * 0.25), int(size * 0.35))
    ], np.int32)
    cv2.polylines(img, [pts_sup], False, vessel_color, 4, cv2.LINE_AA)
    
    # Inferior temporal arcade
    pts_inf = np.array([
        disc_center,
        (int(size * 0.60), int(size * 0.68)),
        (int(size * 0.42), int(size * 0.72)),
        (int(size * 0.25), int(size * 0.65))
    ], np.int32)
    cv2.polylines(img, [pts_inf], False, vessel_color, 4, cv2.LINE_AA)
    
    # Nasal vessels
    cv2.line(img, disc_center, (int(size * 0.88), int(size * 0.35)), vessel_color, 3, cv2.LINE_AA)
    cv2.line(img, disc_center, (int(size * 0.88), int(size * 0.65)), vessel_color, 3, cv2.LINE_AA)
    
    # Smaller branches
    cv2.line(img, (int(size * 0.60), int(size * 0.32)), (int(size * 0.55), int(size * 0.20)), vessel_color, 2, cv2.LINE_AA)
    cv2.line(img, (int(size * 0.42), int(size * 0.28)), (int(size * 0.35), int(size * 0.18)), vessel_color, 2, cv2.LINE_AA)
    cv2.line(img, (int(size * 0.60), int(size * 0.68)), (int(size * 0.55), int(size * 0.80)), vessel_color, 2, cv2.LINE_AA)
    
    # Smooth slight blur on vessels for natural photographic appearance
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    
    # Re-apply circular aperture mask
    img[~mask] = 0
    return img


def add_microaneurysms(img: np.ndarray, count: int = 8) -> np.ndarray:
    """Adds small tiny red capillary outpouchings (microaneurysms)."""
    h, w = img.shape[:2]
    out = img.copy()
    np.random.seed(42)
    for _ in range(count):
        # Place mainly in mid-periphery and macula region
        cx = np.random.randint(int(w * 0.25), int(w * 0.65))
        cy = np.random.randint(int(h * 0.30), int(h * 0.70))
        radius = np.random.randint(2, 4)
        cv2.circle(out, (cx, cy), radius, (90, 10, 10), -1)
    return out


def add_hemorrhages(img: np.ndarray, count: int = 15, large: bool = False) -> np.ndarray:
    """Adds dot and blot retinal hemorrhages."""
    h, w = img.shape[:2]
    out = img.copy()
    np.random.seed(101)
    for _ in range(count):
        cx = np.random.randint(int(w * 0.20), int(w * 0.75))
        cy = np.random.randint(int(h * 0.20), int(h * 0.80))
        r_x = np.random.randint(4, 9 if not large else 16)
        r_y = np.random.randint(3, 7 if not large else 14)
        angle = np.random.randint(0, 180)
        cv2.ellipse(out, (cx, cy), (r_x, r_y), angle, 0, 360, (75, 8, 8), -1)
    return out


def add_hard_exudates(img: np.ndarray, count: int = 25) -> np.ndarray:
    """Adds bright yellowish-white lipid exudate clusters near macula."""
    h, w = img.shape[:2]
    out = img.copy()
    np.random.seed(202)
    # Cluster center near macula
    cluster_cx, cluster_cy = int(w * 0.35), int(h * 0.45)
    for _ in range(count):
        offset_x = int(np.random.normal(0, w * 0.06))
        offset_y = int(np.random.normal(0, h * 0.06))
        cx, cy = cluster_cx + offset_x, cluster_cy + offset_y
        radius = np.random.randint(2, 5)
        # Bright yellowish lipid deposit
        cv2.circle(out, (cx, cy), radius, (245, 240, 150), -1)
    return out


def add_cotton_wool_spots(img: np.ndarray, count: int = 5) -> np.ndarray:
    """Adds fluffy white/grayish nerve fiber layer infarcts (cotton wool spots)."""
    h, w = img.shape[:2]
    out = img.copy()
    np.random.seed(303)
    for _ in range(count):
        cx = np.random.randint(int(w * 0.28), int(w * 0.65))
        cy = np.random.randint(int(h * 0.25), int(h * 0.75))
        # Soft fluffy white spot
        overlay = out.copy()
        cv2.circle(overlay, (cx, cy), np.random.randint(10, 18), (220, 220, 220), -1)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    return out


def add_neovascularization(img: np.ndarray) -> np.ndarray:
    """Adds fragile, abnormal fronds of new vessels (PDR hallmark) across disc and macula."""
    h, w = img.shape[:2]
    out = img.copy()
    np.random.seed(404)
    disc_center = (int(w * 0.70), int(h * 0.50))
    
    # Fine chaotic arbor of delicate vessels emanating from disc
    for _ in range(12):
        pts = [disc_center]
        curr = list(disc_center)
        for step in range(4):
            curr[0] += np.random.randint(-25, -5)
            curr[1] += np.random.randint(-20, 20)
            pts.append(tuple(curr))
        cv2.polylines(out, [np.array(pts, np.int32)], False, (110, 15, 10), 1, cv2.LINE_AA)
        
    # Vitreous pre-retinal hemorrhage (dark meniscus)
    cv2.ellipse(out, (int(w * 0.48), int(h * 0.38)), (int(w * 0.12), int(h * 0.05)), 15, 0, 180, (50, 5, 5), -1)
    return out


def generate_all_sample_grades(output_dir: str = "samples", size: int = 512):
    """Generates and saves fundus images for Grades 0 through 4."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Grade 0: Normal Retina...")
    grade0 = generate_base_retina(size)
    cv2.imwrite(os.path.join(output_dir, "grade_0_normal.png"), cv2.cvtColor(grade0, cv2.COLOR_RGB2BGR))
    
    print("Generating Grade 1: Mild NPDR (Microaneurysms)...")
    grade1 = add_microaneurysms(grade0.copy(), count=10)
    cv2.imwrite(os.path.join(output_dir, "grade_1_mild.png"), cv2.cvtColor(grade1, cv2.COLOR_RGB2BGR))
    
    print("Generating Grade 2: Moderate NPDR (Hemorrhages & Hard Exudates)...")
    grade2 = add_microaneurysms(grade0.copy(), count=15)
    grade2 = add_hemorrhages(grade2, count=12, large=False)
    grade2 = add_hard_exudates(grade2, count=22)
    cv2.imwrite(os.path.join(output_dir, "grade_2_moderate.png"), cv2.cvtColor(grade2, cv2.COLOR_RGB2BGR))
    
    print("Generating Grade 3: Severe NPDR (Extensive Hemorrhages & Cotton Wool Spots)...")
    grade3 = add_hemorrhages(grade0.copy(), count=35, large=True)
    grade3 = add_hard_exudates(grade3, count=30)
    grade3 = add_cotton_wool_spots(grade3, count=6)
    cv2.imwrite(os.path.join(output_dir, "grade_3_severe.png"), cv2.cvtColor(grade3, cv2.COLOR_RGB2BGR))
    
    print("Generating Grade 4: Proliferative DR (Neovascularization & Vitreous Bleed)...")
    grade4 = add_hemorrhages(grade0.copy(), count=25, large=True)
    grade4 = add_hard_exudates(grade4, count=25)
    grade4 = add_neovascularization(grade4)
    cv2.imwrite(os.path.join(output_dir, "grade_4_proliferative.png"), cv2.cvtColor(grade4, cv2.COLOR_RGB2BGR))
    
    print(f"All 5 clinical fundus sample images generated in '{output_dir}'.")


if __name__ == "__main__":
    generate_all_sample_grades()
