from energy import *

def GZ(theta, relative_density, side=1.0):

    rho = 0.5 - np.abs(0.5 - relative_density)
    theta_abs = np.abs(theta)

    x_bg, y_bg = r_bg(theta_abs, rho, side)
    
    if theta < 0:
        GZ_value = -np.cos(theta_abs) * x_bg - np.sin(theta_abs) * y_bg
    else:
        GZ_value = np.cos(theta_abs) * x_bg + np.sin(theta_abs) * y_bg

    return GZ_value

def plot_GZ_vs_angle(relative_densities, degrees=True):
    angles = np.linspace(-0.1, np.pi/4 + 0.1, 1000)

    for rho in relative_densities:
        GZ_values = [GZ(theta, rho) for theta in angles]
        plt.plot(angles*180/np.pi if degrees else angles, GZ_values, label={rho})

    plt.xlabel('Angle (degrees)' if degrees else 'Angle (radians)')
    plt.ylabel('GZ')
    plt.title('GZ vs Angle')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_GZ_vs_angle(relative_densities=[0.278, 0.279, 0.28, 0.281, 0.282, 0.283], degrees=True)