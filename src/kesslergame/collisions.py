# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import math
import matplotlib.pyplot as plt
import numpy as np

def circle_line_collision(line_A: tuple[float, float], line_B: tuple[float, float], center: tuple[float, float], radius: float) -> bool:
    # Check if circle edge is within the outer bounds of the line segment (offset for radius)
    # Not 100% accurate (some false positives) but fast and rare inaccuracies
    x_bounds = [min(line_A[0], line_B[0]) - radius, max(line_A[0], line_B[0]) + radius]
    if center[0] < x_bounds[0] or center[0] > x_bounds[1]:
        return False
    y_bounds = [min(line_A[1], line_B[1]) - radius, max(line_A[1], line_B[1]) + radius]
    if center[1] < y_bounds[0] or center[1] > y_bounds[1]:
        return False

    # calculate side lengths of triangle formed from the line segment and circle center point
    a = math.dist(line_A, center)
    b = math.dist(line_B, center)
    c = math.dist(line_A, line_B)

    # Heron's formula to calculate area of triangle and resultant height (distance from circle center to line segment)
    s = 0.5 * (a + b + c)

    cen_dist = 2.0 / c * math.sqrt(max(0.0, s * (s-a) * (s-b) * (s-c)))

    # If circle distance to line segment is less than circle radius, they are colliding
    return cen_dist < radius

def circle_line_collision_continuous(
    line_A: tuple[float, float],                   # Start point of line segment (bullet head)
    line_B: tuple[float, float],                   # End point of line segment (bullet tail)
    line_vel: tuple[float, float],                 # Velocity (x, y) of the bullet (moves both endpoints)
    circle_center: tuple[float, float],            # Center (x, y) of the asteroid
    circle_vel: tuple[float, float],               # Velocity (x, y) of the asteroid
    circle_radius: float,                          # Asteroid's radius
    delta_time: float,                             # How much time has passed (seconds)
    debug_plot: bool = False,                       # --- DEBUG PLOTTING INSERT ---
    debug_near_miss_margin: float = None           # --- DEBUG NEAR MISS INSERT ---
) -> bool:
    """
    Continuous collision detection for a moving circle and moving line segment in 2D.
    Includes a fast axis-aligned bounding box (AABB) "quick reject" for most frames.
    See detailed comments inside for mathematical explanations.
    """
    # ------------- QUICK REJECTION CHECK -------------
    # Before any expensive math, do a fast AABB overlap check.
    # For continuous collision, we must consider ALL possible positions
    # the objects can occupy during delta_time. To be conservative,
    # we make the bounding boxes cover both the start and end positions
    # of the asteroid (circle) and the swept bullet (line segment). We
    # then expand this by the radius, since the bullet collides with
    # the _edge_ of the asteroid.
    #
    # False negatives cannot happen (no real collisions are missed), but
    # there could be rare false positives, handled by the full algorithm.
    # Compute bullet segment endpoints at t=0 and t=-delta_time
    bullet_head_0 = line_A
    bullet_tail_0 = line_B
    bullet_head_1 = (line_A[0] - line_vel[0] * delta_time, line_A[1] - line_vel[1] * delta_time)
    bullet_tail_1 = (line_B[0] - line_vel[0] * delta_time, line_B[1] - line_vel[1] * delta_time)
    # Asteroid circle center at t=0 and t=-delta_time
    circle_0 = circle_center
    circle_1 = (circle_center[0] - circle_vel[0] * delta_time, circle_center[1] - circle_vel[1] * delta_time)
    # Compute AABB for all four endpoints (min/max x and y)
    x_coords = [bullet_head_0[0], bullet_tail_0[0], bullet_head_1[0], bullet_tail_1[0], circle_0[0], circle_1[0]]
    y_coords = [bullet_head_0[1], bullet_tail_0[1], bullet_head_1[1], bullet_tail_1[1], circle_0[1], circle_1[1]]
    x_min = min(x_coords) - circle_radius
    x_max = max(x_coords) + circle_radius
    y_min = min(y_coords) - circle_radius
    y_max = max(y_coords) + circle_radius
    # If the AABB don't overlap at all, no collision is possible
    if (circle_0[0] < x_min or circle_0[0] > x_max) and (circle_1[0] < x_min or circle_1[0] > x_max):
        return False
    if (circle_0[1] < y_min or circle_0[1] > y_max) and (circle_1[1] < y_min or circle_1[1] > y_max):
        return False
    '''
    Continuous collision detection for a moving circle and a moving line-segment in 2D.
    Mathematical Outline:
    ---------------------
      * Both the line segment (bullet) and circle (asteroid) have constant, independent velocities.
      _At any time t in [-delta_time, 0], the bullet is a segment from line_A + t_line_vel to line_B + t*line_vel
        and the asteroid is a circle centered at (circle_center + t*circle_vel) with radius circle_radius.
      * We parameterize not only t, but also u in [0, 1], where u=0 is at the bullet's tail at time t,
        and u=1 is at the bullet's head. So, each (t, u) specifies a running point along the bullet at time t.
      * To find out if, in the last dt seconds, the moving segment ever touched the moving circle,
        we must find if there is any pair (t, u) in [-dt, 0] x [0, 1] such that the bullet point is inside the circle.
      * This yields an equation:
            (C + Vc _t) - (S + Vs_ t + u*(E-S)) has Euclidean norm <= radius,
        where:
            - C: asteroid initial center
            - Vc: asteroid velocity
            - S: segment tail
            - E: segment head
            - Vs: bullet velocity
            - t: time (negative means past)
            - u: fraction along segment [0, 1]
      * Expanding and grouping, this is a quadratic in t,u, specifically:
            (a0 + a1*t + a2*u)^2 + (b0 + b1*t + b2*u)^2 <= radius^2
        (see below for exact expressions for a0, a1, etc.)
      * The set of (t, u) satisfying this quadratic are points inside an ellipse in (t, u) space.
      * Our rectangle bounding (t, u) (specifically, t in [-delta_time, 0], u in [0, 1]) becomes a parallelogram
        after the affine transformation below.
      * The ellipse can be mapped via affine transform into a unit circle, and the rectangle in (t, u)
        maps to a parallelogram in the transformed space.
      * Collision detection is then reduced to finding if the (origin-centered) circle overlaps the parallelogram.
    Steps:
      1. Build coefficients for the parameterized ellipse (a0, a1, a2, b0, b1, b2, c).
      2. Map the rectangle corners in (t, u) to (t', u') space = parallelogram corners.
      3. Test if any corner is inside the circle.
      4. For each edge of the parallelogram, project the origin onto the edge, clamp to edge;
         if closest point is inside circle, there is intersection.
      5. (Very rare possibility) If the origin itself is strictly inside the parallelogram,
         then the circle is _entirely contained_ and must also return collision (this is commented out).
    '''
    # -------- Helper functions --------
    def dot(a, b):
        """Dot product of 2D vectors."""
        return a[0]*b[0] + a[1]*b[1]
    def dist2(a, b):
        """Squared distance between two points."""
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx*dx + dy*dy
    def dist(a, b):
        """Euclidean distance between two points."""
        return math.sqrt(dist2(a, b))
    # -- 1. Compute bold coefficients for the general quadratic in (t, u) --
    # Note: line_A is bullet head, line_B is bullet tail
    # For parameter u: u=0 is at bullet tail (line_B), u=1 is at bullet head (line_A)
    # So: P_bullet(t, u) = line_B + t*line_vel + u*(line_A - line_B)
    #     P_circle(t)    = circle_center + t*circle_vel
    ax, ay = line_A       # bullet head
    bx, by = line_B       # bullet tail
    vbx, vby = line_vel   # bullet velocity (moves both points)
    cx, cy = circle_center
    vcx, vcy = circle_vel
    ar = circle_radius
    # The vector from the bullet point (at t,u) to asteroid center (at t):
    #   D(t, u) = (cx + vcx*t) - (bx + vbx*t + u*(ax - bx))
    #           = (cx - bx) + (vcx - vbx)*t - u*(ax - bx)
    # Do the same for y.
    a0 = cx - bx
    a1 = vcx - vbx
    a2 = -(ax - bx)
    b0 = cy - by
    b1 = vcy - vby
    b2 = -(ay - by)
    c = ar
    # The collision equation: (a0 + a1*t + a2*u)^2 + (b0 + b1*t + b2*u)^2 = c^2
    # This is an ellipse in (t, u), where we seek (t, u) pairs with t in [-delta_time, 0], u in [0, 1].
    # We'll need to invert the linear system used for the affine change-of-basis from (t, u) to (t', u'):
    # (t', u') = (a0 + a1*t + a2*u, b0 + b1*t + b2*u)
    # In matrix form: |a1 a2| [t] + [a0] = t'
    #                 |b1 b2| [u]   [b0] = u'
    # To solve for (t, u) given (t', u'), use Cramer's rule / matrix inversion:
    #     det = a1*b2 - a2*b1
    #   t = (b2*(t' - a0) - a2*(u' - b0)) / det
    #   u = (a1*(u' - b0) - b1*(t' - a0)) / det
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        # Degenerate case: bullet and asteroid have parallel motion along or across the segment.
        # Can't invert system reliably.
        # -- Fallback: just check for collision at segment start & end for both old and new positions. --
        tail_pos = (bx, by)
        head_pos = (ax, ay)
        tail_pos_prev = (bx - vbx * delta_time, by - vby * delta_time)
        head_pos_prev = (ax - vbx * delta_time, ay - vby * delta_time)
        circle_prev = (cx - vcx * delta_time, cy - vcy * delta_time)
        def point_line_dist(P, A, B):
            # Compute closest point from P to segment AB, return its Euclidean distance.
            ax_, ay_ = A
            bx_, by_ = B
            px, py = P
            abx, aby = bx_ - ax_, by_ - ay_
            apx, apy = px - ax_, py - ay_
            denom = abx*abx + aby*aby
            t = (abx*apx + aby*apy) / denom if denom > 1e-8 else 0
            t = max(0, min(1, t))
            closest = (ax_ + abx * t, ay_ + aby * t)
            return dist(P, closest)
        # Check collision at old and new positions
        last_frame = point_line_dist(circle_prev, tail_pos_prev, head_pos_prev) < ar
        curr_frame = point_line_dist((cx, cy), tail_pos, head_pos) < ar
        return last_frame or curr_frame
    # --------- 2. Parallelogram corners: mapping from parameter rectangle [-dt,0] x [0,1] to (t', u') space ---------
    # The rectangle in (t, u) (parameter: time & segment progress) is:
    #    Lower left  (t, u) = (-delta_time, 0)
    #    Lower right (0, 0)
    #    Upper left  (-delta_time, 1)
    #    Upper right (0, 1)
    corners_tu = [
        (-delta_time, 0),   # Lower left:   t = -delta_time, u = 0
        (0, 0),             # Lower right:  t = 0,           u = 0
        (-delta_time, 1),   # Upper left:   t = -delta_time, u = 1
        (0, 1)              # Upper right:  t = 0,           u = 1
    ]
    corners_tp_up = []
    for t, u in corners_tu:
        # Transform each corner to (t', u') space for collision checks.
        # (This is an affine map, so the rectangle becomes a (possibly skewed) parallelogram.)
        tp = a0 + a1 * t + a2 * u   # x-coordinate in affine-transformed parameter space
        up = b0 + b1 * t + b2 * u   # y-coordinate of same
        corners_tp_up.append( (tp, up) )
    # -------- 3. Check if any parallelogram corner is inside the transformed circle -----------
    #
    # In the transformed space, the original ellipse is now just a circle centered at the origin (0,0)
    # with radius c = circle_radius. The problem reduces to checking if the parallelogram overlaps the origin-centered circle.
    #
    # If any corner of the parallelogram lies within the radius, then there is a collision.
    for tp, up in corners_tp_up:
        if tp*tp + up*up <= c*c:
            # At least one (t,u) pair yields a bullet-colliding-with-asteroid within given delta_time.

            # --- DEBUG PLOTTING INSERT ---
            if debug_plot:
                fig, axs = plt.subplots(1, 3, figsize=(16,5))
                ## 1. World Space
                ax0 = axs[0]
                ax0.plot([ax, bx], [ay, by], color='blue', lw=3, label='Bullet now')
                ax0.plot([ax - vbx * delta_time, bx - vbx * delta_time],
                         [ay - vby * delta_time, by - vby * delta_time],
                         color='blue', lw=2, ls='--', label='Bullet start')
                circ0 = plt.Circle((cx, cy), ar, fill=False, color='r', lw=2)
                circ1 = plt.Circle((cx - vcx * delta_time, cy - vcy * delta_time), ar, fill=False, color='r', ls='--', lw=2)
                ax0.add_patch(circ0)
                ax0.add_patch(circ1)
                ax0.arrow(bx - vbx * delta_time, by - vby * delta_time, vbx * delta_time, vby * delta_time,
                          width=0.02, color='blue', alpha=0.5, label='Bullet movement', length_includes_head=True)
                ax0.arrow(cx - vcx * delta_time, cy - vcy * delta_time, vcx * delta_time, vcy * delta_time,
                          width=0.02, color='red', alpha=0.5, label='Asteroid movement', length_includes_head=True)
                ax0.scatter([ax, bx], [ay, by], c='b')
                ax0.scatter([ax - vbx * delta_time, bx - vbx * delta_time],
                            [ay - vby * delta_time, by - vby * delta_time], c='b', marker='x')
                ax0.scatter([cx], [cy], color='r')
                ax0.scatter([cx - vcx * delta_time], [cy - vcy * delta_time], color='r', marker='x')
                ax0.set_title('World (Physical) Space')
                ax0.set_xlabel('x')
                ax0.set_ylabel('y')
                ax0.legend(fontsize=7)
                ax0.axis('equal')
                ## 2. (t, u) parameter space: rectangle & ellipse
                ax1 = axs[1]
                # Plot the rectangle
                rect = np.array([
                    [-delta_time, 0],
                    [0, 0],
                    [0, 1],
                    [-delta_time, 1],
                    [-delta_time, 0]
                ])
                ax1.plot(rect[:,0], rect[:,1], 'k-', lw=2, label='(t,u) rectangle')
                # Plot the ellipse as a parametric curve for prettiness
                ellipse_points = []
                ellipse_points_u = []
                n_ellipse = 500
                ts = np.linspace(-delta_time*2, delta_time*0.5, n_ellipse)
                ell_angles = np.linspace(0, 2*np.pi, n_ellipse)
                for theta in ell_angles:
                    # Solve for (t,u): Set (a0+a1*t+a2*u)^2 + (b0+b1*t+b2*u)^2 = c^2
                    # Let's parametrize one variable and solve the other.
                    # Here, parametrize u in [0,1] for many t.
                    # Instead, parametrize t along several ellipse points.
                    # For each angle, make a circle in (t',u') of radius c, transform back
                    tprime = c * np.cos(theta)
                    uprime = c * np.sin(theta)
                    denom = det
                    if abs(denom) > 1e-10:
                        t_ = (b2*(tprime - a0) - a2*(uprime - b0)) / denom
                        u_ = (a1*(uprime - b0) - b1*(tprime - a0)) / denom
                        ellipse_points.append([t_, u_])
                ellipse_points = np.array(ellipse_points)
                if len(ellipse_points):
                    ax1.plot(ellipse_points[:,0], ellipse_points[:,1], 'r-', lw=2, label='ellipse')
                ax1.set_title("(t,u) rectangle (black) & ellipse (red)")
                ax1.set_xlabel('t (time)')
                ax1.set_ylabel('u (segment)')
                ax1.set_xlim(-delta_time-0.2, 0.2)
                ax1.set_ylim(-0.2, 1.2)
                ax1.legend(fontsize=7)
                ## 3. Transformed parallelogram and circle
                ax2 = axs[2]
                pparr = np.array([
                    corners_tp_up[0],
                    corners_tp_up[1],
                    corners_tp_up[3],
                    corners_tp_up[2],
                    corners_tp_up[0]
                ])
                ax2.plot(pparr[:,0], pparr[:,1], 'k-', lw=2, label='parallelogram')
                for idx, pt in enumerate(pparr[:-1]): # skip duplicate end point
                    ax2.plot([pt[0]], [pt[1]], 'ko')
                    #ax2.text(pt[0], pt[1], f"{idx}")
                theta = np.linspace(0, 2*np.pi, 300)
                ax2.plot(c*np.cos(theta), c*np.sin(theta), 'r', label='Collision circle')
                ax2.set_title("Transformed parallelogram & circle")
                ax2.set_xlabel("t'")
                ax2.set_ylabel("u'")
                ax2.axis('equal')
                ax2.legend(fontsize=8)
                plt.tight_layout()
                plt.show()
            # --- END DEBUG PLOTTING INSERT ---
            return True
    # -------- 4. For each edge of the parallelogram, check if it crosses or is tangent to the circle. --------
    #
    # For each segment, project the origin (which is the center of the circle in this frame of reference)
    # onto the segment, clamp the projection to the segment, then check if it lies within or on the radius.
    #
    # If so, this edge is intersecting the circle (=> collision event).
    edges = [
        (corners_tp_up[0], corners_tp_up[1]),
        (corners_tp_up[0], corners_tp_up[2]),
        (corners_tp_up[1], corners_tp_up[3]),
        (corners_tp_up[2], corners_tp_up[3])
    ]
    for edge_start, edge_end in edges:
        sx, sy = edge_start
        ex, ey = edge_end
        dx, dy = ex - sx, ey - sy
        seg_len2 = dx*dx + dy*dy
        if seg_len2 < 1e-12:
            continue  # Ignore degenerate (zero-length) edges
        # The vector from the segment start to the origin (parallelogram is in t',u' space)
        # Project the origin onto the segment. 't_proj' is the fraction along the segment.
        # (sx,sy) + t_proj*(dx,dy) = closest point to the origin on this segment.
        # Use vector projection formula.
        t_proj = -(sx*dx + sy*dy) / seg_len2
        t_proj = max(0, min(1, t_proj))  # Clamp to segment endpoints
        closest_x = sx + dx * t_proj
        closest_y = sy + dy * t_proj
        if closest_x*closest_x + closest_y*closest_y <= c*c:
            # --- DEBUG PLOTTING INSERT ---
            if debug_plot:
                fig, axs = plt.subplots(1, 3, figsize=(16,5))
                ax0 = axs[0]
                ax0.plot([ax, bx], [ay, by], color='blue', lw=3, label='Bullet now')
                ax0.plot([ax - vbx * delta_time, bx - vbx * delta_time],
                         [ay - vby * delta_time, by - vby * delta_time],
                         color='blue', lw=2, ls='--', label='Bullet start')
                circ0 = plt.Circle((cx, cy), ar, fill=False, color='r', lw=2)
                circ1 = plt.Circle((cx - vcx * delta_time, cy - vcy * delta_time), ar, fill=False, color='r', ls='--', lw=2)
                ax0.add_patch(circ0)
                ax0.add_patch(circ1)
                ax0.arrow(bx - vbx * delta_time, by - vby * delta_time, vbx * delta_time, vby * delta_time,
                          width=0.02, color='blue', alpha=0.5, label='Bullet movement', length_includes_head=True)
                ax0.arrow(cx - vcx * delta_time, cy - vcy * delta_time, vcx * delta_time, vcy * delta_time,
                          width=0.02, color='red', alpha=0.5, label='Asteroid movement', length_includes_head=True)
                ax0.scatter([ax, bx], [ay, by], c='b')
                ax0.scatter([ax - vbx * delta_time, bx - vbx * delta_time],
                            [ay - vby * delta_time, by - vby * delta_time], c='b', marker='x')
                ax0.scatter([cx], [cy], color='r')
                ax0.scatter([cx - vcx * delta_time], [cy - vcy * delta_time], color='r', marker='x')
                ax0.set_title('World (Physical) Space')
                ax0.set_xlabel('x')
                ax0.set_ylabel('y')
                ax0.legend(fontsize=7)
                ax0.axis('equal')
                ax1 = axs[1]
                rect = np.array([
                    [-delta_time, 0],
                    [0, 0],
                    [0, 1],
                    [-delta_time, 1],
                    [-delta_time, 0]
                ])
                ax1.plot(rect[:,0], rect[:,1], 'k-', lw=2, label='(t,u) rectangle')
                ellipse_points = []
                n_ellipse = 500
                ell_angles = np.linspace(0, 2*np.pi, n_ellipse)
                for theta in ell_angles:
                    tprime = c * np.cos(theta)
                    uprime = c * np.sin(theta)
                    denom = det
                    if abs(denom) > 1e-10:
                        t_ = (b2*(tprime - a0) - a2*(uprime - b0)) / denom
                        u_ = (a1*(uprime - b0) - b1*(tprime - a0)) / denom
                        ellipse_points.append([t_, u_])
                ellipse_points = np.array(ellipse_points)
                if len(ellipse_points):
                    ax1.plot(ellipse_points[:,0], ellipse_points[:,1], 'r-', lw=2, label='ellipse')
                ax1.set_title("(t,u) rectangle (black) & ellipse (red)")
                ax1.set_xlabel('t (time)')
                ax1.set_ylabel('u (segment)')
                #ax1.set_xlim(-delta_time-0.2, 0.2)
                #ax1.set_ylim(-0.2, 1.2)
                ax1.set_aspect('auto')
                ax1.legend(fontsize=7)
                ax2 = axs[2]
                pparr = np.array([
                    corners_tp_up[0],
                    corners_tp_up[1],
                    corners_tp_up[3],
                    corners_tp_up[2],
                    corners_tp_up[0]
                ])
                ax2.plot(pparr[:,0], pparr[:,1], 'k-', lw=2, label='parallelogram')
                for idx, pt in enumerate(pparr[:-1]):
                    ax2.plot([pt[0]], [pt[1]], 'ko')
                    #ax2.text(pt[0], pt[1], f"{idx}")
                theta = np.linspace(0, 2*np.pi, 300)
                ax2.plot(c*np.cos(theta), c*np.sin(theta), 'r', label='Collision circle')
                ax2.set_title("Transformed parallelogram & circle")
                ax2.set_xlabel("t'")
                ax2.set_ylabel("u'")
                ax2.axis('equal')
                ax2.legend(fontsize=8)
                plt.tight_layout()
                plt.show()
            # --- END DEBUG PLOTTING INSERT ---
            return True  # Closest point on edge is within the circle

    # -------- 5. (Optional/rare) Origin strictly inside parallelogram (circle fully inside range): --------
    #
    # It's possible for the parallelogram (in the transformed space) to fully enclose the circle,
    # without any edge or vertex being within the circle. In that case, the origin (center of circle)
    # will be strictly inside the parallelogram.
    #
    # This is an extremely rare situation for normal game parameters (never happens in asteroids),
    # but for mathematical completeness here is the test (commented out):
    #
    # # Helper: For checking if a point is inside a convex quad using cross products (winding).
    # def sign(p1, p2, p3):
    #     # Returns positive if p3 is to the left of the edge p1->p2, negative if to the right.
    #     return (p1[0] - p3[0])*(p2[1] - p3[1]) - (p2[0] - p3[0])*(p1[1] - p3[1])
    # zero = (0, 0)
    # d1 = sign(corners_tp_up[0], corners_tp_up[1], zero)
    # d2 = sign(corners_tp_up[1], corners_tp_up[3], zero)
    # d3 = sign(corners_tp_up[3], corners_tp_up[2], zero)
    # d4 = sign(corners_tp_up[2], corners_tp_up[0], zero)
    # has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
    # has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)
    # if not (has_neg and has_pos):
    #     # All windings are the same sign: origin is strictly inside parallelogram
    #     return True
    #
    # See https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle

    # --- DEBUG NEAR MISS PLOTTING INSERT ---
    if debug_plot and (debug_near_miss_margin is not None):
        t_vals = np.linspace(-delta_time, 0, 200)
        u_vals = np.linspace(0, 1, 100)
        min_dist_value = float('inf')
        min_t = min_u = None
        for t in t_vals:
            for u in u_vals:
                bullet_pt = (bx + vbx*t + (ax - bx)*u, by + vby*t + (ay - by)*u)
                circle_pt = (cx + vcx*t, cy + vcy*t)
                d = math.hypot(bullet_pt[0] - circle_pt[0], bullet_pt[1] - circle_pt[1])
                if d < min_dist_value:
                    min_dist_value = d
                    min_t = t
                    min_u = u
        if (min_dist_value < debug_near_miss_margin) and (min_dist_value > ar):
            print(f"NEAR MISS: Closest approach {min_dist_value:.4f} at t={min_t:.4f}, u={min_u:.4f}")
            fig, axs = plt.subplots(1, 3, figsize=(16,5))
            # 1. World space
            ax0 = axs[0]
            ax0.plot([ax, bx], [ay, by], color='blue', lw=3, label='Bullet now')
            ax0.plot([ax - vbx * delta_time, bx - vbx * delta_time],
                     [ay - vby * delta_time, by - vby * delta_time],
                     color='blue', lw=2, ls='--', label='Bullet start')
            circ0 = plt.Circle((cx, cy), ar, fill=False, color='r', lw=2)
            circ1 = plt.Circle((cx - vcx * delta_time, cy - vcy * delta_time), ar, fill=False, color='r', ls='--', lw=2)
            ax0.add_patch(circ0)
            ax0.add_patch(circ1)
            ax0.arrow(bx - vbx * delta_time, by - vby * delta_time, vbx * delta_time, vby * delta_time,
                      width=0.02, color='blue', alpha=0.5, label='Bullet movement', length_includes_head=True)
            ax0.arrow(cx - vcx * delta_time, cy - vcy * delta_time, vcx * delta_time, vcy * delta_time,
                      width=0.02, color='red', alpha=0.5, label='Asteroid movement', length_includes_head=True)
            ax0.scatter([ax, bx], [ay, by], c='b')
            ax0.scatter([ax - vbx * delta_time, bx - vbx * delta_time],
                        [ay - vby * delta_time, by - vby * delta_time], c='b', marker='x')
            ax0.scatter([cx], [cy], color='r')
            ax0.scatter([cx - vcx * delta_time], [cy - vcy * delta_time], color='r', marker='x')
            # Points of closest approach
            bullet_pt = (bx + vbx*min_t + (ax - bx)*min_u, by + vby*min_t + (ay - by)*min_u)
            circle_pt = (cx + vcx*min_t, cy + vcy*min_t)
            ax0.plot([bullet_pt[0]], [bullet_pt[1]], 'go', markersize=10, label="Closest bullet")
            ax0.plot([circle_pt[0]], [circle_pt[1]], 'mo', markersize=10, label="Closest asteroid")
            ax0.set_title('World (Physical) Space')
            ax0.set_xlabel('x')
            ax0.set_ylabel('y')
            ax0.legend(fontsize=7)
            ax0.axis('equal')
            # 2. (t, u) parameter space
            ax1 = axs[1]
            rect = np.array([
                [-delta_time, 0],
                [0, 0],
                [0, 1],
                [-delta_time, 1],
                [-delta_time, 0]
            ])
            ax1.plot(rect[:,0], rect[:,1], 'k-', lw=2, label='(t,u) rectangle')
            ellipse_points = []
            n_ellipse = 500
            ell_angles = np.linspace(0, 2*np.pi, n_ellipse)
            for theta in ell_angles:
                tprime = c * np.cos(theta)
                uprime = c * np.sin(theta)
                denom = det
                if abs(denom) > 1e-10:
                    t_e = (b2*(tprime - a0) - a2*(uprime - b0)) / denom
                    u_e = (a1*(uprime - b0) - b1*(tprime - a0)) / denom
                    ellipse_points.append([t_e, u_e])
            ellipse_points = np.array(ellipse_points)
            if len(ellipse_points):
                ax1.plot(ellipse_points[:,0], ellipse_points[:,1], 'r-', lw=2, label='ellipse')
            ax1.plot([min_t], [min_u], 'go', label="Closest (t,u)", markersize=10)
            ax1.set_title("(t,u) rectangle (black) & ellipse (red)")
            ax1.set_xlabel('t (time)')
            ax1.set_ylabel('u (segment)')
            #ax1.set_xlim(-delta_time-0.2, 0.2)
            #ax1.set_ylim(-0.2, 1.2)
            ax1.set_aspect('auto')
            ax1.legend(fontsize=7)
            # 3. Transformed parallelogram and circle
            ax2 = axs[2]
            pparr = np.array([
                corners_tp_up[0],
                corners_tp_up[1],
                corners_tp_up[3],
                corners_tp_up[2],
                corners_tp_up[0]
            ])
            ax2.plot(pparr[:,0], pparr[:,1], 'k-', lw=2, label='parallelogram')
            for idx, pt in enumerate(pparr[:-1]): # skip duplicate end point
                ax2.plot([pt[0]], [pt[1]], 'ko')
                #ax2.text(pt[0], pt[1], f"{idx}")
            theta = np.linspace(0, 2*np.pi, 300)
            ax2.plot(c*np.cos(theta), c*np.sin(theta), 'r', label='Collision circle')
            # Transformed closest
            min_tp = a0 + a1 * min_t + a2 * min_u
            min_up = b0 + b1 * min_t + b2 * min_u
            ax2.plot([min_tp], [min_up], 'go', markersize=10, label="Closest t',u'")
            ax2.set_title("Transformed parallelogram & circle")
            ax2.set_xlabel("t'")
            ax2.set_ylabel("u'")
            ax2.axis('equal')
            ax2.legend(fontsize=8)
            plt.tight_layout()
            plt.show()
    # --- END DEBUG NEAR MISS PLOTTING INSERT ---

    # ---------- No collision detected in the current frame interval -----------
    return False
