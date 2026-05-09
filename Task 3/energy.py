import numpy as np
import matplotlib.pyplot as plt


def _rotate_square_vertices(theta, side=1.0):
    half = side / 2.0
    corners = [

        (-half, -half),
        (half, -half),
        (half, half),
        (-half, half),
    ]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return [
        (x * cos_t - y * sin_t, x * sin_t + y * cos_t)
        for x, y in corners
    ]


def _clip_polygon_to_waterline(polygon):
    clipped = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        inside1 = y1 <= 0.0
        inside2 = y2 <= 0.0

        if inside1:
            clipped.append((x1, y1))

        if inside1 ^ inside2:
            t = y1 / (y1 - y2)
            x_int = x1 + t * (x2 - x1)
            clipped.append((x_int, 0.0))

    return clipped


def _polygon_area_and_centroid(polygon):
    if len(polygon) < 3:
        return 0.0, (0.0, 0.0)

    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(polygon)

    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    area *= 0.5
    if abs(area) < 1e-16:
        return 0.0, (0.0, 0.0)

    cx /= 6.0 * area
    cy /= 6.0 * area
    return abs(area), (cx, cy)


def _submerged_area_and_centroid(theta, y0, side=1.0):
    vertices = _rotate_square_vertices(theta, side)
    translated = [(x, y + y0) for x, y in vertices]
    submerged = _clip_polygon_to_waterline(translated)
    return _polygon_area_and_centroid(submerged)


def equilibrium_centroid_vertical_offset(theta, relative_density, side=1.0, tol=1e-10, max_iter=100):
    if relative_density <= 0.0 or relative_density >= 1.0:
        raise ValueError("relative_density must be between 0 and 1 for a floating prism")

    target_area = relative_density * side * side
    lower = -side / 2.0
    upper = side / 2.0
    f_lower = _submerged_area_and_centroid(theta, lower, side)[0] - target_area
    f_upper = _submerged_area_and_centroid(theta, upper, side)[0] - target_area

    if f_lower < 0 or f_upper > 0:
        raise RuntimeError("Unable to bracket the equilibrium immersion depth")

    for _ in range(max_iter):
        mid = 0.5 * (lower + upper)
        area_mid = _submerged_area_and_centroid(theta, mid, side)[0]
        f_mid = area_mid - target_area
        if abs(f_mid) < tol or abs(upper - lower) < tol:
            return mid
        if f_mid > 0:
            lower = mid
        else:
            upper = mid

    return 0.5 * (lower + upper)


def potential_energy(theta, relative_density, side=1.0, g=9.81):
    theta_array = np.atleast_1d(theta)
    scalar_input = theta_array.shape == ()
    energies = []

    for angle in np.nditer(theta_array):
        y0 = equilibrium_centroid_vertical_offset(float(angle), relative_density, side)
        _, centroid = _submerged_area_and_centroid(float(angle), y0, side)
        y_b = centroid[1]
        energies.append(g * relative_density * side * side * (y0 - y_b))

    energies = np.array(energies)
    if scalar_input:
        return float(energies)
    return energies.reshape(theta_array.shape)


def find_equilibrium_angles(relative_density, side=1.0, g=9.81, angle_range=(-np.pi/2, np.pi/2), num_points=2000, tol=1e-8):
    """
    Find angles where potential energy has local minima (stable equilibria).
    Returns angles in radians where the potential is locally minimal.
    """
    angles = np.linspace(angle_range[0], angle_range[1], num_points)
    energies = potential_energy(angles, relative_density, side, g)
    dtheta = angles[1] - angles[0]

    stable_equilibria = []
    for i in range(1, len(angles) - 1):
        if energies[i] <= energies[i - 1] + tol and energies[i] <= energies[i + 1] + tol:
            if energies[i] + tol < energies[i - 1] or energies[i] + tol < energies[i + 1]:
                # Refine minimum location with quadratic interpolation
                y0, y1, y2 = energies[i - 1], energies[i], energies[i + 1]
                denom = (y0 - 2 * y1 + y2)
                if abs(denom) > tol:
                    offset = 0.5 * (y0 - y2) / denom
                else:
                    offset = 0.0
                eq_angle = angles[i] + offset * dtheta

                # Check local curvature for stability
                V_plus = potential_energy(eq_angle + dtheta, relative_density, side, g)
                V_minus = potential_energy(eq_angle - dtheta, relative_density, side, g)
                V_center = potential_energy(eq_angle, relative_density, side, g)
                d2V_dtheta2 = (V_plus - 2 * V_center + V_minus) / (dtheta ** 2)
                if d2V_dtheta2 > 0:
                    stable_equilibria.append(eq_angle)

    # Include endpoints as minima only when they are lower than the adjacent point
    if len(angles) > 1:
        if energies[0] < energies[1]:
            stable_equilibria.append(angles[0])
        if energies[-1] < energies[-2]:
            stable_equilibria.append(angles[-1])

    # Remove duplicates within tolerance and sort
    unique_equilibria = []
    for angle in sorted(stable_equilibria):
        if not unique_equilibria or abs(angle - unique_equilibria[-1]) > tol:
            unique_equilibria.append(angle)

    return np.array(unique_equilibria)


def _variable_density_grid(density_range, num_densities, high_density_ranges=None, high_weight=2.0, low_weight=0.5):
    """Create a 1D density grid with finer sampling in selected subranges."""
    if high_density_ranges is None:
        high_density_ranges = [(0.2, 0.3), (0.7, 0.8)]

    density_range = (max(density_range[0], 0.0), min(density_range[1], 1.0))
    segments = []
    current = density_range[0]

    for lo, hi in sorted(high_density_ranges):
        lo = max(lo, density_range[0])
        hi = min(hi, density_range[1])
        if lo >= hi:
            continue
        if current < lo:
            segments.append((current, lo, low_weight))
        segments.append((lo, hi, high_weight))
        current = hi

    if current < density_range[1]:
        segments.append((current, density_range[1], low_weight))

    total_weight = sum((b - a) * w for a, b, w in segments)
    densities = []
    for i, (a, b, w) in enumerate(segments):
        count = max(2, int(round(num_densities * (b - a) * w / total_weight)))
        if i < len(segments) - 1:
            segment_points = np.linspace(a, b, count, endpoint=False)
        else:
            segment_points = np.linspace(a, b, count, endpoint=True)
        densities.append(segment_points)

    if densities:
        densities = np.concatenate(densities)
    else:
        densities = np.array([])

    return np.unique(densities)


def bifurcation_diagram(density_range=(0.1, 0.9), num_densities=50, side=1.0, g=9.81):
    """
    Compute bifurcation diagram showing equilibrium angles vs density.
    Returns densities and corresponding equilibrium angles.
    """
    densities = _variable_density_grid(density_range, num_densities)
    all_equilibria = []

    for rho in densities:
        eq_angles = find_equilibrium_angles(rho, side, g, angle_range=(-np.pi/4, np.pi/4))
        all_equilibria.append(eq_angles)

    return densities, all_equilibria


def plot_bifurcation_diagram(density_range=(0.1, 0.9), num_densities=100, side=1.0, g=9.81, degrees=True, show=True, save_path=None):
    """
    Plot bifurcation diagram showing equilibrium paths for changing density.
    """
    densities, all_equilibria = bifurcation_diagram(density_range, num_densities, side, g)

    fig, ax = plt.subplots(figsize=(5, 6))

    # Plot each equilibrium point as a discrete marker
    for rho, eq_angles in zip(densities, all_equilibria):
        if len(eq_angles) > 0:
            x_vals = np.degrees(eq_angles) if degrees else eq_angles
            y_vals = np.full_like(x_vals, rho)
            ax.plot(x_vals, y_vals, 'o', color='tab:blue', markersize=4, alpha=0.7)

    ax.set_xlabel('Equilibrium Angle (degrees)' if degrees else 'Equilibrium Angle (radians)')
    ax.set_ylabel('Relative Density')
    ax.set_title('Bifurcation Diagram: Equilibrium Angles vs Density')
    ax.grid(True, alpha=0.3)

    # Set reasonable axis limits
    if degrees:
        ax.set_xlim(-50, 50)
    else:
        ax.set_xlim(-np.pi/3.6, np.pi/3.6)

    ax.set_ylim(density_range[0], density_range[1])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def example_energy_curve(relative_density=0.6, side=1.0, g=9.81, num_points=181):
    angles = np.linspace(-np.pi / 4.0, np.pi / 4.0, num_points)  # -45° to +45°
    energies = potential_energy(angles, relative_density, side, g)
    return angles, energies


def plot_potential_vs_angle(relative_density, side=1.0, g=9.81, num_points=181, degrees=False, show=True, save_path=None):
    """Plot potential energy versus angle for one relative density."""
    if not (0.0 < relative_density < 1.0):
        raise ValueError("relative_density must be between 0 and 1")

    complement = 1.0 - relative_density
    if abs(relative_density - complement) > 1e-12:
        print(
            f"Note: for this square model, relative densities {relative_density:.3f} "
            f"and {complement:.3f} produce identical energy curves."
        )

    angles, energies = example_energy_curve(relative_density, side, g, num_points)
    x = angles * 180.0 / np.pi if degrees else angles
    x_label = "Angle (degrees)" if degrees else "Angle (radians)"

    fig, ax = plt.subplots()
    ax.plot(x, energies, marker="o", markersize=4, linestyle="-", color="tab:blue")
    ax.set_title(f"Potential Energy vs. Angle (density={relative_density}, centered on 0°)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Potential Energy (J/m)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_multiple_potential_vs_angle(relative_densities, side=1.0, g=9.81, num_points=50, degrees=False, normalize_mean=False, show=True, save_path=None):
    """Plot potential energy curves for several relative densities on the same axes."""
    fig, ax = plt.subplots()
    for rho in relative_densities:
        angles, energies = example_energy_curve(rho, side, g, num_points)
        if normalize_mean:
            energies = energies - np.mean(energies)
        x = angles * 180.0 / np.pi if degrees else angles
        ax.plot(x, energies, marker="o", markersize=4, linestyle="-", label=rho)

    x_label = "Angle (degrees)" if degrees else "Angle (radians)"
    y_label = "Normalized Potential Energy (J/m)" if normalize_mean else "Potential Energy (J/m)"
    title = "Mean-Centered Potential Energy vs. Angle for multiple densities" if normalize_mean else "Potential Energy vs. Angle for multiple densities"
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    angles, energies = example_energy_curve(relative_density=0.6)
    for angle, energy in zip(angles[::30], energies[::30]):
        print(f"theta={angle:.3f} rad, energy={energy:.6f} J/m")

    # Test equilibrium finding
    print("\nTesting equilibrium finding for density=0.6:")
    eq_angles = find_equilibrium_angles(0.6)
    print(f"Equilibrium angles: {np.degrees(eq_angles)} degrees")

    # Plot diagrams
    plot_multiple_potential_vs_angle(relative_densities=[0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3], degrees=True, normalize_mean=True)
    #plot_bifurcation_diagram(density_range=(0.05, 0.95), num_densities=100)
