import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def ingest_data(data_file, x_ind, y_ind, y_base, scale_factor, exposure_time, exposure_time_error, measurement_error):
    """
    Reads in data from the csv file and returns a pandas Dataframe with d_y vs velocity data -- with error!
    Args:
        data_file: the csv file containing the data. This should be formatted as follows:
            x1a, y1a
            x1b, y1b
            ...
            xna, ynb

            where consecutive pairs of coordinates denote the endpoints of a streak
        x_ind: the index of the x-coordinate in the csv file
        y_ind: the index of the y-coordinate in the csv file
        y_base: the base of the channel
        scale_factor: the scale factor to apply to go from pixels -> desired unit
        exposure_time: the exposure time of the camera in ms
        measurement_error: a qualitative estimate of the measurement error in pixels
    Returns:
        a pd.DataFrame containing the data in the form:
            d_y, length, velocity, err_y, err_length, err_velocity
    """
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        raw_data = list(reader)
        raw_data.pop(0)  # remove header
    res = []
    err = measurement_error * scale_factor
    for row in range(0, len(raw_data), 2):
        xa = float(raw_data[row][x_ind])
        xb = float(raw_data[row+1][x_ind])
        ya = float(raw_data[row][y_ind])
        yb = float(raw_data[row+1][y_ind])

        # take average of ya, yb because people don't always draw perfectly horizontal lines
        y = (ya + yb) / 2
        d_y = abs(y - y_base) * scale_factor
        length = abs(xb - xa) * scale_factor
        velocity = length / exposure_time
        err_velocity = velocity * ((((err/length)**2) + ((exposure_time_error/exposure_time)**2)) ** 0.5)
        res.append((d_y, length, velocity, err, err, err_velocity))
    return pd.DataFrame(res, columns=['d_y', 'length', 'velocity', 'err_y', 'err_length', 'err_velocity'])




def runner(data_file, x_ind, y_ind, scale_factor, exposure_time, exposure_time_error, measurement_error):
    data = ingest_data(data_file, x_ind, y_ind, y_base, scale_factor, exposure_time, exposure_time_error, measurement_error)
    print(data)
    # take a quadratic fit
    fit, cov = np.polyfit(data['d_y'], data['velocity'], 2, cov=True)
    fn = np.poly1d(fit)
    xs = np.linspace(min(data.d_y), max(data.d_y), 1000)

    p1 = plt.errorbar(data.velocity, data.d_y, xerr=data.err_velocity, yerr = data.err_y, fmt='o', label="Velocity data")
    p2 = plt.plot(fn(xs), xs, label=f"fit: x = {fit[0]:.4f}y^2 + {fit[1]:.4f}y + {fit[2]:.4f}")
    plt.xlabel('Velocity (um/s)')
    plt.ylabel('Distance from base (um)')
    plt.legend()
    plt.title("Velocity Profile")
    plt.show()


if __name__ == '__main__':
    data_file = 'data/straight/straight-p1-10x-25.5-12+-0.5cm-DATA.csv'
    x_ind = 1
    y_ind = 2
    y_base = 200
    scale_factor = 2.155 # pixels -> desired unit (um)
    exposure_time = 25.5 # ms
    exposure_time_error = 0.1
    measurement_error = 5
    runner(data_file, x_ind, y_ind, scale_factor, exposure_time, exposure_time_error, measurement_error)







