"""Rectify the CRT playfield of the source video to a flat 320x236 image (quad from the bright region)."""
import cv2, numpy as np
W, H = 320, 236
def quad(f):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV); m = (hsv[..., 2] > 110).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea)).reshape(-1, 2)
    s = hull.sum(1); d = np.diff(hull, axis=1).ravel()
    return np.float32([hull[s.argmin()], hull[d.argmin()], hull[s.argmax()], hull[d.argmax()]])
def rectify(f, q=None):
    q = quad(f) if q is None else q
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    return cv2.warpPerspective(f, M, (W, H), flags=cv2.INTER_AREA)
