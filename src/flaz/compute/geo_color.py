import math
import colorsys

SE = (333060.9, 7394752.2)  # Sé (CRS métrico)


def geo_color_from_point(
    x: float,
    y: float,
    *,
    max_dist: float = 15_000,
    mode: str = "hex",
):
    dx = x - SE[0]
    dy = y - SE[1]

    # Direção → Hue
    angle = math.atan2(dy, dx)
    hue = (math.degrees(angle) + 360) % 360

    # Distância normalizada
    dist = math.hypot(dx, dy)
    d = min(dist / max_dist, 1.0)

    # 🎯 Saturação quase constante (identidade cromática)
    saturation = 0.75 + 0.10 * (d ** 0.5)
    # varia pouco: 0.75 → ~0.85

    # 🌑 → 🌕 Value cresce forte com a distância
    value = 0.30 + 0.70 * (d ** 0.9)
    # centro ~0.30 (escuro)
    # borda ~1.00 (claro)

    r, g, b = colorsys.hsv_to_rgb(
        hue / 360,
        saturation,
        value
    )

    if mode == "rgb":
        return int(r * 255), int(g * 255), int(b * 255)

    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255),
        int(g * 255),
        int(b * 255)
    )
