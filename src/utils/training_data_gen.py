
import numpy as np
import xarray as xr
from typing import Tuple, Optional
import warnings
from scipy.interpolate import RegularGridInterpolator

class WindFieldDataGenerator:
    '''
    DataGenerator that uses a vector field to generate (x_0, x_1) pairs. Used to train flow matching model.
    '''
    def __init__(self, lats, lons, u_wind, v_wind, dt=0.1, n_steps=100):
        """
        Initialize the generator with a wind field.
        
        Args:
            lats: array of latitude values
            lons: array of longitude values
            u_wind: 2D array of u wind component (east-west)
            v_wind: 2D array of v wind component (north-south)
            dt: time step for integration
            n_steps: number of steps to integrate for each trajectory
        """
        self.lats = lats  # [-90, 90]
        self.lons = lons  # [0, 360]
        self.u_wind = u_wind
        self.v_wind = v_wind
        self.dt = dt
        self.n_steps = n_steps

        # Create interpolators for smooth wind field
        self.u_interp = RegularGridInterpolator(
            (lats, lons), 
            u_wind,
            bounds_error=False,
            fill_value=None
        )
        self.v_interp = RegularGridInterpolator(
            (lats, lons), 
            v_wind,
            bounds_error=False,
            fill_value=None
        )

    def _get_wind_at_point(self, lat, lon):
        lon = lon % 360
        pos = np.array([[lat, lon]])
        return (
            float(self.u_interp(pos)),
            float(self.v_interp(pos))
        )
    
    
    def generate_trajectory(self, lat, lon):
        """
        Generate a single trajectory starting from given lat/lon.
        Uses RK4 integration for better accuracy.
        Return only start and end points
        """

        # Initialize trajectory
        current_lat = lat
        current_lon = lon
        
        # Integrate forward
        for _ in range(self.n_steps):
            dlat, dlon = self._rk4_step(current_lat, current_lon)
            current_lat += dlat
            current_lon += dlon
            current_lat, current_lon = self._wrap_lat_lon(current_lat, current_lon)
        return (lat, lon), (current_lat, current_lon)

    def _wrap_lat_lon(self, lat, lon):
        # Clip longitude to be within 0 and 360
        lon = np.clip(lon, 0, 360)
        # Wrap longitude around the globe
        # lon = lon % 360

        # Wrap latitude around the globe
        if lat > 90:
            lat = 180 - lat
        elif lat < -90:
            lat = -180 - lat
        return lat, lon
    
    def generate_full_trajectory(self, start_lat, start_lon):
        """
        Generate a full trajectory including intermediate points.
        
        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
        
        Returns:
            trajectory: List of (lat, lon) pairs for entire trajectory
        """

        trajectory = [(start_lat, start_lon % 360)]
        current_lat = start_lat
        current_lon = start_lon % 360
        
        for _ in range(self.n_steps):
            dlat, dlon = self._rk4_step(current_lat, current_lon)
            current_lat += dlat
            current_lon += dlon
            current_lat, current_lon = self._wrap_lat_lon(current_lat, current_lon)
            trajectory.append((current_lat, current_lon))
        
        return np.array(trajectory)
    
    def _rk4_step(self, lat, lon):
        u1, v1 = self._get_wind_at_point(lat, lon)
        # These are derivatives, not steps --> remember to multiply by dt
        k1_lat = v1
        k1_lon = u1/np.cos(np.radians(lat))

        u2, v2 = self._get_wind_at_point(lat + self.dt * k1_lat / 2 , lon + self.dt * k1_lon / 2)
        k2_lat = v2
        k2_lon = u2/np.cos(np.radians(lat + self.dt *  v1 / 2))

        u3, v3 = self._get_wind_at_point(lat + self.dt * k2_lat / 2 , lon + self.dt * k2_lon / 2)
        k3_lat = v3
        k3_lon = u3/np.cos(np.radians(lat + self.dt * k2_lat / 2))

        u4, v4 = self._get_wind_at_point(lat + self.dt * k3_lat, lon +  self.dt * k3_lon)
        k4_lat = v4
        k4_lon = u4/np.cos(np.radians(lat + self.dt * k3_lat))

        dlat = self.dt * (k1_lat + 2*k2_lat + 2*k3_lat + k4_lat) / 6
        dlon = self.dt * (k1_lon + 2*k2_lon + 2*k3_lon + k4_lon) / 6

        return dlat, dlon


def create_generator_from_grib(
    grib_path: str,
    pressure_level: Optional[float] = None,
    dt: float = 0.1,
    n_steps: int = 100,
    backend_kwargs: Optional[dict] = None
) -> Tuple[WindFieldDataGenerator, xr.Dataset]:
    """
    Create a WindFieldDataGenerator from a GRB2 file.
    
    Args:
        grib_path: Path to the .grb2 file
        pressure_level: Pressure level in hPa to use. If None, uses first available level
        dt: Time step for trajectory integration
        n_steps: Number of steps for each trajectory
        backend_kwargs: Additional arguments to pass to cfgrib
    
    Returns:
        generator: Configured WindFieldDataGenerator
        ds: The opened dataset for reference
    """

    # Set up default backend kwargs
    if backend_kwargs is None:
        backend_kwargs = {
            'filter_by_keys': {
                'typeOfLevel': 'isobaricInhPa',
                'shortName': ['u', 'v']  # Only load wind components
            },
            'errors': 'ignore'  # Ignore warnings about skipped variables
        }
    
    # Suppress warnings about skipped variables
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(
            grib_path,
            engine='cfgrib',
            backend_kwargs=backend_kwargs
        )

    # Verify we have the necessary variables
    required_vars = ['u', 'v']
    missing_vars = [var for var in required_vars if var not in ds.variables]
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")


    # Select pressure level
    if pressure_level is None:
        pressure_level = ds.isobaricInhPa.values[0]
    elif pressure_level not in ds.isobaricInhPa.values:
        raise ValueError(
            f"Pressure level {pressure_level} not found. Available levels: {ds.isobaricInhPa.values}"
        )
    
    # Extract wind components at the selected pressure level
    u_wind = ds['u'].sel(isobaricInhPa=pressure_level).squeeze()
    v_wind = ds['v'].sel(isobaricInhPa=pressure_level).squeeze()
    
    # Create generator
    generator = WindFieldDataGenerator(
        lats=ds.latitude.values,
        lons=ds.longitude.values,
        u_wind=u_wind.values,
        v_wind=v_wind.values,
        dt=dt,
        n_steps=n_steps
    )
    
    print(f"Created generator for pressure level {pressure_level} hPa")
    print(f"Grid resolution: {len(ds.latitude)}x{len(ds.longitude)}")
    print(f"Latitude range: [{float(ds.latitude.min())}, {float(ds.latitude.max())}]")
    print(f"Longitude range: [{float(ds.longitude.min())}, {float(ds.longitude.max())}]")
    
    return generator, ds

if __name__ == "__main__":
    PATH = './data/gfs_4_20100808_1200_000.grb2'
    generator, ds = create_generator_from_grib(PATH, pressure_level=1000, n_steps=1000)
    trajectory = generator.generate_full_trajectory(-30, 300)
    print(trajectory[:100])