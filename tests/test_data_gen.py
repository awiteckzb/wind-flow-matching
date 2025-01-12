import numpy as np
import xarray as xr

from src.utils.training_data_gen import create_generator_from_grib


def main():
    FILE_PATH = './data/gfs_4_20100808_1200_000.grb2'

    generator, ds = create_generator_from_grib(
        FILE_PATH,
        pressure_level=1000,
        dt=0.1,
        n_steps=100
    )


    # ds = xr.open_dataset(FILE_PATH, 
    #                     engine='cfgrib', 
    #                     filter_by_keys={'typeOfLevel': 'isobaricInhPa'})
    
    # if pressure_level is None:
    #     pressure_level = ds.isobaricInhPa.values[0]
    # elif pressure_level not in ds.isobaricInhPa.values:
    #     raise ValueError(
    #         f"Pressure level {pressure_level} not found. Available levels: {ds.isobaricInhPa.values}"
    #     )
    
    # u_wind = ds['u'].sel(isobaricInhPa=pressure_level).squeeze()
    # v_wind = ds['v'].sel(isobaricInhPa=pressure_level).squeeze()

    # generator = WindFieldDataGenerator(
    #     lats=ds.latitude.values,
    #     lons=ds.longitude.values,
    #     u_wind=u_wind.values,
    #     v_wind=v_wind.values,
    #     dt=dt,
    #     n_steps=n_steps
    # )

    start_lat = np.random.uniform(-80, 80)  # Avoid poles
    start_lon = np.random.uniform(-180, 180)

    start, end = generator.generate_trajectory(start_lat, start_lon)

    print(start, end)

if __name__ == "__main__":
    main()