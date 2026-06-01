# Auto-Modell hier ablegen

Lege dein F1-Auto-Modell als **`f1.glb`** (oder `f1.gltf`) in diesen Ordner:

```
web/public/models/f1.glb
```

Die Szene lädt es automatisch und ersetzt damit den Platzhalter (ein farbiger
Kegel). Bis die Datei existiert, wird der Platzhalter angezeigt — die App läuft
also auch ohne Modell.

Hinweise:
- Format **glTF/GLB** (three.js `GLTFLoader`).
- Das Modell wird automatisch auf ~8 m Länge skaliert und liegt flach in der
  XY-Ebene (Vogelperspektive).
- Zeigt das Auto in die falsche Richtung, passe `MODEL_YAW_OFFSET` in
  `src/three/TrackScene.ts` an (z. B. `Math.PI / 2`).
