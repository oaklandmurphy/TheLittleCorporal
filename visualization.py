
import math
import pygame
from unit import Unit
from pygame.locals import *
from OpenGL.GL import (
    glBegin, glEnd, glColor3f, glVertex2f, glClearColor, glClear, glDrawPixels, glWindowPos2d,
    GL_POLYGON, GL_LINE_LOOP, GL_QUADS, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINES, glLineWidth, GL_LINE_STRIP
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
from map.terrain import FIELDS, FOREST, RIVER, HILL

# Visual configuration
WINDOW_W = 1000
WINDOW_H = 700
HEX_SIZE = 36  # radius of hex (distance center -> vertex)
HEX_WIDTH = HEX_SIZE * 2
HEX_HEIGHT = math.sqrt(3) * HEX_SIZE
HEX_HORIZ = HEX_WIDTH * 3/4  # horizontal spacing
HEX_VERT = HEX_HEIGHT        # vertical spacing

def elevation_to_color(elevation: int) -> tuple:
    """
    Convert elevation to RGB color.
    Low elevation (e.g. -0.2) -> light green (0.65, 0.85, 0.6)
    High elevation (e.g. 1.0) -> muddy brown (0.55, 0.45, 0.3)
    """
    # Define color range
    low_color = (0.65, 0.85, 0.6)    # light green
    high_color = (0.55, 0.45, 0.3)   # muddy brown
    
    # Normalize elevation to 0-1 range
    # Assuming elevation typically ranges from -0.2 (rivers) to 1.0 (hills)
    min_elev = 0
    max_elev = 5
    normalized = (elevation - min_elev) / (max_elev - min_elev)
    normalized = max(0.0, min(1.0, normalized))  # clamp to [0,1]
    
    # Linear interpolation between colors
    r = low_color[0] + (high_color[0] - low_color[0]) * normalized
    g = low_color[1] + (high_color[1] - low_color[1]) * normalized
    b = low_color[2] + (high_color[2] - low_color[2]) * normalized
    
    return (r, g, b)


def draw_single_tree(cx: float, cy: float, size: float):
    """Draw a single tree icon."""
    glColor3f(0.15, 0.4, 0.15)  # Dark green
    
    # Draw a simple triangle for tree canopy
    icon_size = size * 0.4
    glBegin(GL_POLYGON)
    glVertex2f(cx, cy - icon_size * 0.6)           # top
    glVertex2f(cx - icon_size * 0.5, cy + icon_size * 0.3)  # bottom left
    glVertex2f(cx + icon_size * 0.5, cy + icon_size * 0.3)  # bottom right
    glEnd()
    
    # Draw trunk as small rectangle
    trunk_width = icon_size * 0.15
    trunk_height = icon_size * 0.4
    glColor3f(0.3, 0.2, 0.1)  # Brown
    glBegin(GL_QUADS)
    glVertex2f(cx - trunk_width/2, cy + icon_size * 0.3)
    glVertex2f(cx + trunk_width/2, cy + icon_size * 0.3)
    glVertex2f(cx + trunk_width/2, cy + icon_size * 0.6)
    glVertex2f(cx - trunk_width/2, cy + icon_size * 0.6)
    glEnd()


def draw_forest_icons(cx: float, cy: float, size: float, tree_cover: int):
    """Draw tree icons based on tree_cover value (0-3 trees)."""
    if tree_cover <= 0:
        return
    
    # Position offsets for multiple trees based on tree_cover
    # These offsets position trees within the hex
    if tree_cover == 1:
        # Single tree in center
        positions = [(0, 0)]
    elif tree_cover == 2:
        # Two trees side by side
        spacing = size * 0.3
        positions = [(-spacing/2, 0), (spacing/2, 0)]
    elif tree_cover >= 3:
        # Three trees in triangle formation
        spacing = size * 0.25
        positions = [
            (0, -spacing * 0.4),           # top
            (-spacing * 0.6, spacing * 0.4),  # bottom left
            (spacing * 0.6, spacing * 0.4)    # bottom right
        ]
    
    # Draw each tree at its position
    for dx, dy in positions:
        draw_single_tree(cx + dx, cy + dy, size * 0.8)  # Slightly smaller when multiple


def draw_river_icon(cx: float, cy: float, size: float):
    """Draw wavy lines to represent a river on top of a hex."""
    glColor3f(0.2, 0.45, 0.8)  # Blue
    glLineWidth(3)
    
    icon_size = size * 0.5
    glBegin(GL_LINE_STRIP)
    # Draw a simple wavy line
    segments = 8
    for i in range(segments + 1):
        t = i / segments
        x = cx - icon_size/2 + icon_size * t
        # Create wave effect using sine
        y = cy + math.sin(t * math.pi * 2) * icon_size * 0.15
        glVertex2f(x, y)
    glEnd()
    
    glLineWidth(1)  # Reset


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


def draw_combat_indicator(x1, y1, x2, y2):
    """Draw a red crossed rifles symbol at the midpoint between two engaged units."""
    cx1, cy1 = hex_to_pixel(x1, y1)
    cx2, cy2 = hex_to_pixel(x2, y2)
    
    # Midpoint between the two hexes
    mid_x = (cx1 + cx2) / 2
    mid_y = (cy1 + cy2) / 2
    
    # Size of the combat indicator
    size = HEX_SIZE * 0.4
    
    # Draw red crossed rifles
    glColor3f(0.9, 0.1, 0.1)  # Red color
    glLineWidth(4)
    glBegin(GL_LINES)
    # First rifle (diagonal from bottom-left to top-right)
    glVertex2f(mid_x - size/2, mid_y + size/2)
    glVertex2f(mid_x + size/2, mid_y - size/2)
    # Second rifle (diagonal from top-left to bottom-right)
    glVertex2f(mid_x - size/2, mid_y - size/2)
    glVertex2f(mid_x + size/2, mid_y + size/2)
    glEnd()
    glLineWidth(1)  # Reset to default


class Visualization:
    def __init__(self, game_map: Map):
        self.game_map = game_map
        self.font = pygame.font.SysFont("Arial", 18)
        self.status_font = pygame.font.SysFont("Arial", 24, bold=True)
        self._background_cache = None  # Will store (width, height, pixel data)
        self._background_cache_valid = False


    def render(self, hover_info=None, llm_processing=False):
        glClearColor(0.92, 0.92, 0.92, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw cached background (static map) if available, else redraw and cache
        if not self._background_cache_valid or self._background_cache is None:
            self._background_cache = self._draw_static_map_to_surface()
            self._background_cache_valid = True

        # Blit the cached background using OpenGL
        bg_width, bg_height, bg_pixels = self._background_cache
        glWindowPos2d(0, 0)
        glDrawPixels(bg_width, bg_height, GL_RGBA, GL_UNSIGNED_BYTE, bg_pixels)

        # Highlight reachable hexes if present in hover_info
        reachable_hexes = None
        if isinstance(hover_info, dict) and "reachable_hexes" in hover_info:
            reachable_hexes = hover_info["reachable_hexes"]

        # Draw overlays (reachable hexes highlight)
        if reachable_hexes:
            from OpenGL.GL import glEnable, glDisable, glBlendFunc, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            for (col, row) in reachable_hexes:
                hex = self.game_map.get_hex(col, row)
                if hex is None:
                    continue
                terrain = hex.terrain
                base_color = elevation_to_color(terrain.elevation)
                # Overlay a blue highlight (mix color)
                color = tuple(min(1.0, c + 0.35) if i == 2 else c for i, c in enumerate(base_color))
                cx, cy = hex_to_pixel(col, row)
                # Draw with 50% opacity
                from OpenGL.GL import glColor4f, glBegin, glEnd, glVertex2f, GL_POLYGON, GL_LINE_LOOP
                glColor4f(color[0], color[1], color[2], 0.5)
                glBegin(GL_POLYGON)
                for (x, y) in hex_polygon_points(cx, cy, HEX_SIZE):
                    glVertex2f(x, y)
                glEnd()
                # outline
                glColor4f(0.05, 0.05, 0.05, 0.5)
                glBegin(GL_LINE_LOOP)
                for (x, y) in hex_polygon_points(cx, cy, HEX_SIZE):
                    glVertex2f(x, y)
                glEnd()
            glDisable(GL_BLEND)

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

        # Draw combat indicators for engaged units
        self.draw_combat_indicators()

        # Draw overlay info
        if hover_info:
            # Only pass lines to render_hover_info
            lines = hover_info["lines"] if isinstance(hover_info, dict) and "lines" in hover_info else hover_info
            self.render_hover_info(lines)

        # Draw LLM processing indicator
        if llm_processing:
            self.render_llm_processing_indicator()

    def _draw_static_map_to_surface(self):
        """Draw the static map (terrain, forests, rivers) to a Pygame surface and return (w, h, pixel data)."""
        # Create a Pygame surface with alpha channel
        surface = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA, 32)
        surface = surface.convert_alpha()
        # Draw terrain and features to the surface using Pygame drawing
        for row in range(self.game_map.height):
            for col in range(self.game_map.width):
                hex = self.game_map.get_hex(col, row)
                if hex is None:
                    continue
                terrain = hex.terrain
                color = tuple(int(255 * c) for c in elevation_to_color(terrain.elevation)) + (255,)
                cx, cy = hex_to_pixel(col, row)
                # Draw filled hex polygon
                points = hex_polygon_points(cx, cy, HEX_SIZE)
                pygame.draw.polygon(surface, color, points)
                # Draw hex outline
                pygame.draw.polygon(surface, (13, 13, 13, 255), points, 1)
                # Draw forest icons
                if terrain.tree_cover > 0:
                    # Draw using Pygame for caching; fallback to a simple circle for each tree
                    for i in range(terrain.tree_cover):
                        # Offset positions for up to 3 trees
                        if terrain.tree_cover == 1:
                            offsets = [(0, 0)]
                        elif terrain.tree_cover == 2:
                            spacing = HEX_SIZE * 0.3
                            offsets = [(-spacing/2, 0), (spacing/2, 0)]
                        else:
                            spacing = HEX_SIZE * 0.25
                            offsets = [(0, -spacing * 0.4), (-spacing * 0.6, spacing * 0.4), (spacing * 0.6, spacing * 0.4)]
                        ox, oy = offsets[i]
                        pygame.draw.circle(surface, (38, 102, 38, 255), (int(cx + ox), int(cy + oy)), int(HEX_SIZE * 0.18))
                # Draw river icon (simple blue line)
                if terrain.name == "River":
                    pygame.draw.line(surface, (51, 115, 204, 255), (int(cx - HEX_SIZE * 0.3), int(cy)), (int(cx + HEX_SIZE * 0.3), int(cy)), 4)
        # Convert surface to string buffer for OpenGL
        pixel_data = pygame.image.tostring(surface, "RGBA", True)
        return surface.get_width(), surface.get_height(), pixel_data

    def invalidate_background_cache(self):
        """Call this when the map changes (terrain/features) to force background redraw."""
        self._background_cache_valid = False

    def get_hex_at_pixel(self, px, py):
        """
        Convert pixel coordinates to hex coordinates using proper hexagonal geometry.
        This accounts for the actual shape of hexagons, not just rectangular approximation.
        """
        px_adj = px - 60
        py_adj = py - 40
        
        # Rough estimate of which hex we might be in
        col_estimate = px_adj / HEX_HORIZ
        
        # Check candidate hexes (the estimated one and neighbors)
        # We need to check multiple candidates because edges overlap
        col_candidates = [int(math.floor(col_estimate)), int(math.ceil(col_estimate))]
        
        best_hex = None
        best_dist = float('inf')
        
        for col in col_candidates:
            if col < 0 or col >= self.game_map.width:
                continue
                
            # Calculate y offset for odd columns
            y_offset = (HEX_VERT / 2) if (col % 2) else 0
            row_estimate = (py_adj - y_offset) / HEX_VERT
            
            # Check multiple row candidates
            row_candidates = [int(math.floor(row_estimate)), int(math.ceil(row_estimate))]
            
            for row in row_candidates:
                if row < 0 or row >= self.game_map.height:
                    continue
                
                # Get the center of this hex
                hex_cx, hex_cy = hex_to_pixel(col, row)
                
                # Calculate distance from mouse to hex center
                dist = math.sqrt((px - hex_cx)**2 + (py - hex_cy)**2)
                
                # If this is the closest hex center so far, and it's within range
                if dist < best_dist and dist < HEX_SIZE:
                    best_dist = dist
                    best_hex = (col, row)
        
        return best_hex

    def get_hover_info(self, mouse_pos):
        """Return a dict with lines describing the hovered hex/unit, and reachable hexes if a unit is present."""
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
        terrain_line = f"{t.name} (Move: {t.move_cost}, Elev: {t.elevation}, Trees: {t.tree_cover})"

        reachable_hexes = None
        if hex.unit:
            unit_line = hex.unit.status() if hasattr(hex.unit, "status") else str(hex.unit)
            lines.append(unit_line)
            lines.append(f"Terrain: {terrain_line}")
            # Compute reachable hexes for this unit
            reachable_hexes = set(self.game_map.find_reachable_hexes(hex.unit).keys())
        else:
            lines.append(terrain_line)

        if getattr(hex, "features", None):
            feat_str = "; ".join(hex.features)
            if feat_str:
                lines.append(f"Features: {feat_str}")

        result = {"lines": lines}
        if reachable_hexes:
            result["reachable_hexes"] = reachable_hexes
        return result

    def render_hover_info(self, hover_info):
        if not hover_info:
            return
        
        # Check if processing indicator is set
        processing = False
        lines = []
        
        if isinstance(hover_info, dict):
            processing = hover_info.get("processing", False)
            # Get lines from dict
            if "lines" in hover_info:
                lines = hover_info["lines"]
            elif len(hover_info) > 1 or not processing:
                # Dict contains other data besides processing flag
                lines = [str(v) for k, v in hover_info.items() if k != "processing"]
        elif isinstance(hover_info, str):
            lines = hover_info.splitlines()
        elif isinstance(hover_info, list):
            lines = hover_info
        else:
            lines = []

        # Add processing indicator at the top
        if processing:
            lines.insert(0, "⏳ AI Thinking...")

        if not lines:
            return

        x = 20
        y = 20
        line_gap = 4
        for i, line in enumerate(lines):
            # Use yellow background for processing indicator
            bg_color = (255, 255, 180) if i == 0 and processing else (255, 255, 220)
            text_color = (150, 80, 0) if i == 0 and processing else (20, 20, 20)
            text_surface = self.font.render(line, True, text_color, bg_color)
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

    def draw_combat_indicators(self):
        """Draw red crossed rifles symbols between engaged units."""
        processed_pairs = set()
        
        for row in range(self.game_map.height):
            for col in range(self.game_map.width):
                hex = self.game_map.get_hex(col, row)
                if not hex or not hex.unit or not hex.unit.engagement:
                    continue
                
                unit = hex.unit
                
                # Check for adjacent engaged enemies
                for nx, ny in self.game_map.get_neighbors(col, row):
                    if not (0 <= nx < self.game_map.width and 0 <= ny < self.game_map.height):
                        continue
                    
                    neighbor_hex = self.game_map.get_hex(nx, ny)
                    if not neighbor_hex or not neighbor_hex.unit:
                        continue
                    
                    enemy = neighbor_hex.unit
                    
                    # Check if this is an engaged enemy pair
                    if enemy.faction != unit.faction and enemy.engagement:
                        # Create a sorted tuple to avoid drawing the same pair twice
                        pair = tuple(sorted([(col, row), (nx, ny)]))
                        
                        if pair not in processed_pairs:
                            processed_pairs.add(pair)
                            draw_combat_indicator(col, row, nx, ny)

