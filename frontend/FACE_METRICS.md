# How the face metrics work

The interview page reports three things per answer: eye contact %, head movement and
blink rate. All three are computed in the browser, in [lib/faceMetrics.ts](lib/faceMetrics.ts),
from the output of MediaPipe's Face Landmarker. No video leaves the browser.

## What MediaPipe gives us

For every video frame, about 30 times a second:

1. **478 landmarks.** Fixed points on the face (landmark 1 is always the nose tip, 33 is
   always the outer corner of one eye, and so on). Each has `x` and `y` between 0 and 1,
   as a fraction of the image width and height. Landmarks 468-477 are the two irises.
2. **A 4x4 transformation matrix.** It describes how a standard "face looking straight at
   the camera" has to be rotated and moved to match the face in the frame.

MediaPipe does the hard part (finding the face). Everything below is geometry on those
numbers.

One detail matters throughout: `x` and `y` are normalised separately, so on a 640x480
frame a step of 0.1 in `x` is 64 pixels but in `y` only 48. Every distance is therefore
computed after multiplying by the frame width and height.

## 1. Blink rate: the eye aspect ratio

Take six landmarks around one eye:

```
        p2    p3
   p1            p4        p1, p4 = corners
        p6    p5           p2, p3 = upper lid, p6, p5 = lower lid
```

```
EAR = ( |p2 - p6| + |p3 - p5| ) / ( 2 * |p1 - p4| )
```

That is eye height divided by eye width. An open eye gives about 0.3. As the lids close,
the two vertical distances shrink to nearly 0 while the width stays the same, so EAR
drops toward 0. Because it is a ratio, moving closer to or further from the camera does
not change it.

We average the EAR of both eyes and count a blink with **two thresholds**:

- EAR falls below 0.20 → the eye is now "closed"
- EAR rises above 0.24 while "closed" → the eye is "open" again, count one blink

Why two thresholds instead of one? With a single cut-off at 0.22, an EAR hovering at
0.21, 0.23, 0.21, 0.23 from noise would count as two blinks. The gap between the
thresholds (hysteresis) means the eye has to clearly close and clearly reopen.

```
blinks per minute = blinks / (tracked seconds / 60)
```

People blink roughly 10-20 times a minute at rest. It tends to rise under stress and
fall when staring at a screen.

## 2. Head pose: reading the rotation out of the matrix

The upper-left 3x3 block of the matrix is a rotation matrix `R`. Its columns are where
the face's own axes end up in camera space. The **third column** is the direction the
face is pointing:

```
forward = (fx, fy, fz) = (m[8], m[9], m[10])      (the array is stored column by column)
```

Facing the camera exactly, `forward = (0, 0, 1)`. From it we get two angles:

```
yaw   = atan2(fx, fz)                  turning left / right
pitch = atan2(-fy, sqrt(fx² + fz²))    nodding up / down
```

`atan2(fx, fz)` is the angle of the forward vector in the horizontal plane. For pitch,
`sqrt(fx² + fz²)` is how much of the vector lies in that plane and `fy` is how much
points up or down, so their ratio gives the tilt angle.

**Head movement** is how fast those angles change:

```
per frame:  change = sqrt( Δyaw² + Δpitch² )          degrees
head movement = sum of changes / tracked seconds      degrees per second
```

Raw landmarks jitter by a fraction of a degree on every frame even when the head is
still. At 30 frames a second that jitter alone would add up to several degrees per
second. So the angles are smoothed first with an exponential moving average:

```
smoothed = previous + 0.3 * (new - previous)
```

Each new value only moves the estimate 30% of the way, which damps frame-to-frame noise
but still follows real movement within a few frames.

## 3. Eye contact

A frame counts as eye contact when both of these hold:

**The head faces the camera:** `|yaw| < 15°` and `|pitch| < 15°`.

**The irises are centred in the eyes.** For each eye, project the iris centre onto the
line between the eye's two corners:

```
t = ( (iris - left) · (right - left) ) / |right - left|²
```

The dot product divided by the squared length gives the position along the line as a
fraction: 0 at the left corner, 1 at the right corner, 0.5 in the middle. Projecting
(rather than just comparing `x` values) keeps this correct when the head is tilted. We
average `t` over both eyes and require `|t - 0.5| < 0.15`.

Both conditions are needed. Head pose alone misses someone facing the camera while
reading notes off to the side; iris position alone misses someone whose head is turned
away with their eyes centred.

```
eye contact % = frames with eye contact / frames where a face was found * 100
```

## Limits worth knowing

- **Camera vs screen.** A webcam sits a few centimetres above the screen. Looking at the
  interviewer's face on screen and looking into the lens differ by only a few degrees,
  which is inside the 15° tolerance. So this measures "looking toward the camera and
  screen", not strictly into the lens.
- **Vertical gaze is not measured from the iris.** Eyelids cover part of the iris, which
  makes its vertical position unreliable, so up/down relies on head pitch only.
- **Thresholds are fixed.** 15°, 0.15 and the EAR cut-offs are reasonable defaults, not
  calibrated per person. Glasses, narrow eyes or a low camera can shift the numbers.
- **Lighting and frame rate.** In poor light the landmarks get noisier, and a blink
  (about 150 ms) can fall between frames on a slow machine and be missed.
