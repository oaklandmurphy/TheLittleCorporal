
import math
import pygame
from unit import Unit
from pygame.locals import *
from OpenGL.GL import (
    glBegin, glEnd, glColor3f, glVertex2f, glClearColor, glClear, glDrawPixels, glWindowPos2d,
    GL_POLYGON, GL_LINE_LOOP, GL_QUADS, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINES, glLineWidth
)
def draw_unit_square(unit: Unit):
    """Draw a fallback rectangle for non-infantry units."""
    if unit.x is None or unit.y is None:
        return
    cx, cy = hex_to_pixel(unit.x, unit.y)
    width = HEX_SIZE * 0.5
    height = HEX_SIZE * 0.5
    color = FACTION_COLORS.get(unit.faction, (1.0, 1.0, 1.0))
    glColor3f(*color)
    glBegin(GL_QUADS)
    glVertex2f(cx - width/2, cy - height/2)
    glVertex2f(cx + width/2, cy - height/2)
    glVertex2f(cx + width/2, cy + height/2)
    glVertex2f(cx - width/2, cy + height/2)
    glEnd()
    glColor3f(0.0, 0.0, 0.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(cx - width/2, cy - height/2)
    glVertex2f(cx + width/2, cy - height/2)
    glVertex2f(cx + width/2, cy + height/2)
    glVertex2f(cx - width/2, cy + height/2)
    glEnd()
from OpenGL.GLU import gluOrtho2D
import sys

# Import your game model classes (adjust imports if you placed them in a package)
from map import Map        # Map, Hex hexs with .terrain and .unit
from unit import Unit     # Unit class (or specific subclasses)
from terrain import FIELDS, FOREST, RIVER, HILL

# Visual configuration
WINDOW_W = 1000
WINDOW_H = 700
HEX_SIZE = 36  # radius of hex (distance center -> vertex)
HEX_WIDTH = HEX_SIZE * 2
HEX_HEIGHT = math.sqrt(3) * HEX_SIZE
HEX_HORIZ = HEX_WIDTH * 3/4  # horizontal spacing
HEX_VERT = HEX_HEIGHT        # vertical spacing

# Colors for terrains (R,G,B) floats 0..1
TERRAIN_COLORS = {
    "Fields": (0.65, 0.85, 0.6),
    "Forest": (0.2, 0.5, 0.2),
    "River":  (0.2, 0.45, 0.8),
    "Hill":   (0.6, 0.5, 0.35)
}
# Fallback color
DEFAULT_TERRAIN_COLOR = (0.6, 0.6, 0.6)

# Faction colors
FACTION_COLORS = {
    "French": (0.2, 0.3, 0.7),
    "Austrian":  (0.95, 0.77, 0.22),
    "Russian": (0.2, 0.7, 0.1),
    "Prussian": (0.1, 0.1, 0.1),
    "British": (0.9, 0.3, 0.3)
}

def hex_to_pixel(col: int, row: int):
    """
    Convert odd-q offset coordinates (column, row) to pixel coordinates.
    col = x, row = y.
    Uses the same odd-q assumptions as Map.get_neighbors.
    Returns pixel center (px, py).
    """
    px = col * HEX_HORIZ + 60  # small margin offset
    # odd columns are shifted down by HEX_HEIGHT/2
    py = row * HEX_VERT + (HEX_VERT / 2 if col % 2 else 0) + 40
    return px, py

def hex_polygon_points(cx: float, cy: float, size: float):
    """
    Return list of 6 (x,y) vertices for a pointy-topped hex centered at (cx,cy).
    Start angle -30 degrees so hex is pointy-topped.
    """
    points = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = math.radians(angle_deg)
        x = cx + size * math.cos(angle_rad)
        y = cy + size * math.sin(angle_rad)
        points.append((x, y))
    return points

def draw_hex(cx: float, cy: float, color):
    """Draw filled hex polygon at center (cx,cy) with given color tuple."""
    glColor3f(*color)
    glBegin(GL_POLYGON)
    for (x, y) in hex_polygon_points(cx, cy, HEX_SIZE):
        glVertex2f(x, y)
    glEnd()
    # outline
    glColor3f(0.05, 0.05, 0.05)
    glBegin(GL_LINE_LOOP)
    for (x, y) in hex_polygon_points(cx, cy, HEX_SIZE):
        glVertex2f(x, y)
    glEnd()

def draw_infantry_symbol(unit: Unit):
    """Draw a NATO infantry symbol: horizontal rectangle with crossed diagonal lines."""
    if unit.x is None or unit.y is None:
        return
    cx, cy = hex_to_pixel(unit.x, unit.y)
    width = HEX_SIZE * 1.0
    height = HEX_SIZE * 0.7
    color = FACTION_COLORS.get(unit.faction, (1.0, 1.0, 1.0))
    # Draw filled rectangle
    glColor3f(*color)
    glBegin(GL_QUADS)
    glVertex2f(cx - width/2, cy - height/2)
    glVertex2f(cx + width/2, cy - height/2)
    glVertex2f(cx + width/2, cy + height/2)
    glVertex2f(cx - width/2, cy + height/2)
    glEnd()
    # Draw border
    glColor3f(0.0, 0.0, 0.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(cx - width/2, cy - height/2)
    glVertex2f(cx + width/2, cy - height/2)
    glVertex2f(cx + width/2, cy + height/2)
    glVertex2f(cx - width/2, cy + height/2)
    glEnd()
    # Draw crossed diagonal lines (draw after rectangle and border)
    margin = 1  # pixels inside the rectangle
    glColor3f(0.0, 0.0, 0.0)
    glLineWidth(3)
    glBegin(GL_LINES)
    glVertex2f(cx - width/2 + margin, cy - height/2 + margin)
    glVertex2f(cx + width/2 - margin, cy + height/2 - margin)
    glVertex2f(cx + width/2 - margin, cy - height/2 + margin)
    glVertex2f(cx - width/2 + margin, cy + height/2 - margin)
    glEnd()
    glLineWidth(1)  # Reset to default


class Visualization:
    def __init__(self, game_map: Map):
        self.game_map = game_map
        self.font = pygame.font.SysFont("Arial", 18)
        self.status_font = pygame.font.SysFont("Arial", 24, bold=True)

    def render(self, hover_info=None, llm_processing=False):
        glClearColor(0.92, 0.92, 0.92, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Draw hexes
        for row in range(self.game_map.height):
            for col in range(self.game_map.width):
                hex = self.game_map.get_hex(col, row)
                if hex is None:
                    continue
                terrain = hex.terrain
                color = TERRAIN_COLORS.get(terrain.name, DEFAULT_TERRAIN_COLOR)
                cx, cy = hex_to_pixel(col, row)
                draw_hex(cx, cy, color)
        # Draw units on top
        from unit import Infantry
        for row in range(self.game_map.height):
            for col in range(self.game_map.width):
                hex = self.game_map.get_hex(col, row)
                if hex and hex.unit:
                    # If unit is infantry, draw NATO symbol
                    if isinstance(hex.unit, Infantry):
                        draw_infantry_symbol(hex.unit)
                    else:
                        draw_unit_square(hex.unit)  # fallback for other unit types
        # Draw overlay info
        if hover_info:
            self.render_hover_info(hover_info)
        
        # Draw LLM processing indicator
        if llm_processing:
            self.render_llm_processing_indicator()

    def get_hex_at_pixel(self, px, py):
        # Improved odd-q offset hex picking
        px_adj = px - 60
        py_adj = py - 40
        if px_adj < 0 or py_adj < 0:
            return None
        col = int(round(px_adj / HEX_HORIZ))
        # odd columns are shifted down by HEX_HEIGHT/2
        y_offset = (HEX_VERT / 2) if (col % 2) else 0
        row = int(round((py_adj - y_offset) / HEX_VERT))
        if col < 0 or row < 0:
            return None
        return (col, row)

    def get_hover_info(self, mouse_pos):
        """Return a list of lines describing the hovered hex/unit, including feature names and coordinates."""
        hex_coords = self.get_hex_at_pixel(*mouse_pos)
        if not hex_coords:
            return None
        col, row = hex_coords
        hex = self.game_map.get_hex(col, row)
        if not hex:
            return None

        lines = []
        # Show coordinates first
        lines.append(f"Hex: ({col}, {row})")

        t = hex.terrain
        terrain_line = f"{t.name} (Move: {t.move_cost}, Combat: {getattr(t, 'combat_modifier', getattr(t, 'combat_mod', 1.0))})"
        
        if hex.unit:
            unit_line = hex.unit.status() if hasattr(hex.unit, "status") else str(hex.unit)
            lines.append(unit_line)
            lines.append(f"Terrain: {terrain_line}")
        else:
            lines.append(terrain_line)

        if getattr(hex, "features", None):
            feat_str = "; ".join(hex.features)
            if feat_str:
                lines.append(f"Features: {feat_str}")

        return lines if lines else None

    def render_hover_info(self, hover_info):
        if not hover_info:
            return
        # Normalize to list of lines
        if isinstance(hover_info, str):
            lines = hover_info.splitlines()
        else:
            lines = list(hover_info)

        x = 20
        y = 20
        line_gap = 4
        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, (20, 20, 20), (255, 255, 220))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glWindowPos2d(x, y + i * (self.font.get_height() + line_gap))
            glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    def render_llm_processing_indicator(self):
        """Draw a processing indicator to show LLM is working."""
        import time
        # Animated dots
        dots = int(time.time() * 2) % 4
        status_text = "LLM Processing" + "." * dots
        
        text_surface = self.status_font.render(status_text, True, (255, 100, 0), (50, 50, 50))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        
        # Position in top-right corner
        x = WINDOW_W - text_surface.get_width() - 20
        y = 20
        
        glWindowPos2d(x, y)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
