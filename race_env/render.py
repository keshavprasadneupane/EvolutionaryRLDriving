"""pygame renderer shared by SingleCarRace and MultiCarRace (render_mode="human").

It draws purely from data the engine hands out:
  * `get_static()`      -> track bands / grass bounds / walls / checkpoints
  * `get_observation()` -> car position + angle (car outline comes from CarConfig)
  * `cast_view()`       -> lidar rays (toggle with L)

Keys: L lidar, Tab next car, mouse wheel zoom, Esc / window close.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

# colors
BACKGROUND = (38, 38, 42)
GRASS = (46, 110, 56)
ROAD = (92, 92, 98)
WALL = (140, 96, 60)
START_LINE = (240, 240, 240)
CHECKPOINT = (240, 150, 40)
CAR_COLORS = [
    (225, 70, 70), (70, 150, 230), (240, 200, 60), (170, 90, 220),
    (70, 200, 160), (240, 140, 60), (200, 200, 200), (230, 100, 170),
]
# lidar end dot by SurfaceType: Track, Grass, Wall, Outside
RAY_END = [(0, 220, 230), (240, 150, 40), (240, 220, 60), (255, 255, 255)]

LAYER_WALL_OUTER, LAYER_GRASS_OUTER, LAYER_TRACK, LAYER_GRASS_INNER, LAYER_WALL_INNER = range(5)


def _band_quads(points: np.ndarray, normal: np.ndarray, half: float) -> np.ndarray:
    """(N, 4, 2) quads between sample i-1 and i (same corners as TrackMap::bandQuad)."""
    pp, pn = np.roll(points, 1, axis=0), np.roll(normal, 1, axis=0)
    return np.stack(
        [pp - pn * half, pp + pn * half, points + normal * half, points - normal * half],
        axis=1,
    ).astype(np.float32)


class PygameRenderer:
    def __init__(self, car_vertices: Sequence[Sequence[float]], fps: int = 60,
                 size: tuple[int, int] = (1100, 760), title: str = "race_env"):
        try:
            import pygame
        except ImportError as e:  # pragma: no cover
            raise ImportError("render_mode='human' needs pygame: pip install pygame") from e
        self.pg = pygame
        pygame.display.init()  # only what is needed (no audio mixer)
        pygame.font.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 22)
        self.fps = fps
        self.zoom = size[1] / 90.0  # pixels per world unit
        self.focus = 0
        self.show_lidar = False
        self.closed = False

        self.car_vertices = np.asarray(car_vertices, dtype=np.float32).reshape(-1, 2)
        # the two front-most vertices make the "headlight" line
        self._front = np.argsort(self.car_vertices[:, 1])[-2:]

        self._static_key: Optional[int] = None
        self._bands: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = []
        self._static: Optional[dict] = None

    # ------------------------------------------------------------------ static
    def set_static(self, static: dict) -> None:
        """Precompute the track polygons. Call again after a new track."""
        self._static = static
        pts, nrm, hw = static["layers"], static["normal"], static["layer_half_widths"]
        self._bands = []
        if static["has_walls"]:
            order = [(LAYER_GRASS_OUTER, GRASS), (LAYER_GRASS_INNER, GRASS),
                     (LAYER_TRACK, ROAD), (LAYER_WALL_OUTER, WALL), (LAYER_WALL_INNER, WALL)]
        else:  # grass is one big rectangle (drawn separately), only the road is a band
            order = [(LAYER_TRACK, ROAD)]
        for layer, color in order:
            q = _band_quads(pts[layer], nrm, float(hw[layer]))
            self._bands.append((q, q.mean(axis=1), color))

    # --------------------------------------------------------------------- draw
    def _to_screen(self, p: np.ndarray, cam: np.ndarray, w: int, h: int) -> np.ndarray:
        s = np.empty_like(p, dtype=np.float32)
        s[..., 0] = (p[..., 0] - cam[0]) * self.zoom + w / 2
        s[..., 1] = h / 2 - (p[..., 1] - cam[1]) * self.zoom
        return s

    def _car_polygon(self, pos: np.ndarray, angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        v = self.car_vertices
        return np.stack([c * v[:, 0] - s * v[:, 1] + pos[0],
                         s * v[:, 0] + c * v[:, 1] + pos[1]], axis=1)

    def _handle_events(self, n_cars: int) -> None:
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.closed = True
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE:
                    self.closed = True
                elif e.key == pg.K_l:
                    self.show_lidar = not self.show_lidar
                elif e.key == pg.K_TAB and n_cars > 0:
                    self.focus = (self.focus + 1) % n_cars
            elif e.type == pg.MOUSEWHEEL:
                self.zoom = float(np.clip(self.zoom * (1.1 ** e.y), 0.5, 60.0))

    def draw(self, static: dict, observations: Sequence, hud: Sequence[str] = (),
             cast_view: Optional[Callable[[int], dict]] = None, tick: bool = True) -> None:
        """Draw one frame. `observations` = list of CarObservation (one per car)."""
        if self.closed:
            return
        pg = self.pg
        if self._static is not static and id(static) != self._static_key:
            self.set_static(static)
            self._static_key = id(static)

        self._handle_events(len(observations))
        if self.closed:
            return
        self.focus = min(self.focus, len(observations) - 1)

        surf = pg.display.get_surface()
        W, H = surf.get_size()
        surf.fill(BACKGROUND)
        cam = np.asarray(observations[self.focus].position, dtype=np.float32)

        # grass rectangle (no walls): fills the bounds the engine reports
        if not static["has_walls"]:
            x0, y0, x1, y1 = (float(v) for v in static["bounds"])
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            pg.draw.polygon(surf, GRASS, self._to_screen(corners, cam, W, H).tolist())

        # track bands, only those near the camera
        view_r = math.hypot(W, H) / 2 / self.zoom
        pad = 3.0 * float(static["sample_step"]) + float(static["layer_half_widths"].max())
        for quads, centers, color in self._bands:
            near = np.hypot(centers[:, 0] - cam[0], centers[:, 1] - cam[1]) < view_r + pad
            for q in self._to_screen(quads[near], cam, W, H):
                pg.draw.polygon(surf, color, q.tolist())

        # start line + checkpoints
        c0 = static["centerline"][0]
        n0 = static["normal"][0]
        hw = float(static["track_half_width"])
        a = self._to_screen(np.stack([c0 - n0 * hw, c0 + n0 * hw]), cam, W, H)
        pg.draw.line(surf, START_LINE, a[0].tolist(), a[1].tolist(), 4)
        for inner, outer in zip(static["checkpoint_inner"], static["checkpoint_outer"]):
            a = self._to_screen(np.stack([inner, outer]), cam, W, H)
            pg.draw.line(surf, CHECKPOINT, a[0].tolist(), a[1].tolist(), 2)

        # cars
        for i, o in enumerate(observations):
            poly = self._to_screen(self._car_polygon(np.asarray(o.position), o.angle), cam, W, H)
            color = CAR_COLORS[i % len(CAR_COLORS)]
            pg.draw.polygon(surf, color, poly.tolist())
            pg.draw.polygon(surf, (255, 255, 255) if i == self.focus else (20, 20, 20),
                            poly.tolist(), 3 if i == self.focus else 1)
            f = poly[self._front]
            pg.draw.line(surf, (255, 245, 150), f[0].tolist(), f[1].tolist(), 3)

        # lidar of the followed car
        if self.show_lidar and cast_view is not None:
            rays = cast_view(self.focus)
            s = self._to_screen(rays["start"], cam, W, H)
            e = self._to_screen(rays["end"], cam, W, H)
            for k in range(len(s)):
                pg.draw.line(surf, (230, 60, 60) if rays["hit"][k] else (90, 220, 90),
                             s[k].tolist(), e[k].tolist(), 1)
                pg.draw.circle(surf, RAY_END[int(rays["surface"][k])], e[k].tolist(), 4)

        # HUD
        lines = list(hud) + ["L lidar   Tab next car   wheel zoom"]
        for k, text in enumerate(lines):
            surf.blit(self.font.render(text, True, (255, 255, 255)), (10, 8 + 20 * k))

        pg.display.flip()
        if tick:
            self.clock.tick(self.fps)

    def close(self) -> None:
        self.pg.display.quit()
        self.pg.font.quit()
        self.closed = True