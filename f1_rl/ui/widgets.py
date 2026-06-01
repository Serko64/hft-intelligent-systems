"""Tiny mouse-friendly UI widgets (buttons + checkboxes).

These are *immediate-mode* helpers: a view decides where a widget goes (a
pygame.Rect) and calls draw_button / draw_checkbox every frame. The same rect
is used by app.py to test mouse clicks. There is no widget state to manage —
the caller owns the on/off value and the layout.

Usage pattern:
    rect = pygame.Rect(x, y, w, h)
    draw_button(screen, font, rect, "Start", hovered=rect.collidepoint(mouse))
    # ...and in the event loop:
    if event.button == 1 and rect.collidepoint(event.pos):
        do_the_thing()
"""
from __future__ import annotations

import pygame

from f1_rl.config import ACCENT, DIM, GOLD, GREEN, HOV_BG, TEXT


def draw_button(screen, font, rect: pygame.Rect, label: str, *,
                hovered: bool = False, enabled: bool = True,
                primary: bool = False) -> None:
    """A clickable button. `primary` makes it stand out (red accent); `enabled`
    False greys it out (caller should also ignore clicks on it)."""
    if not enabled:
        bg, border, fg = (26, 26, 34), (48, 48, 58), (95, 95, 105)
    elif primary:
        bg = (120, 40, 40) if hovered else (80, 28, 28)
        border, fg = ACCENT, (255, 240, 240)
    else:
        bg = HOV_BG if hovered else (30, 30, 42)
        border = GOLD if hovered else DIM
        fg = TEXT if hovered else (200, 200, 210)

    pygame.draw.rect(screen, bg, rect, border_radius=8)
    pygame.draw.rect(screen, border, rect, 2, border_radius=8)
    lbl = font.render(label, True, fg)
    screen.blit(lbl, (rect.centerx - lbl.get_width() // 2,
                      rect.centery - lbl.get_height() // 2))


def draw_checkbox(screen, font, rect: pygame.Rect, label: str, checked: bool, *,
                  hovered: bool = False) -> None:
    """A labelled checkbox. The whole `rect` is the click target (so clicking
    the label toggles too); the tick box is drawn at the left edge."""
    box = pygame.Rect(rect.x, rect.y + (rect.height - 22) // 2, 22, 22)
    pygame.draw.rect(screen, (28, 28, 42), box, border_radius=5)
    border = GREEN if checked else (GOLD if hovered else DIM)
    pygame.draw.rect(screen, border, box, 2, border_radius=5)
    if checked:
        # hand-drawn checkmark (no font glyph needed)
        pygame.draw.lines(screen, GREEN, False, [
            (box.x + 4, box.centery + 1),
            (box.centerx - 1, box.bottom - 5),
            (box.right - 3, box.y + 4),
        ], 3)
    fg = TEXT if (checked or hovered) else DIM
    lbl = font.render(label, True, fg)
    screen.blit(lbl, (box.right + 10, rect.y + (rect.height - lbl.get_height()) // 2))
