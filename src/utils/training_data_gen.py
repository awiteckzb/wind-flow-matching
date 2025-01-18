
import numpy as np
import xarray as xr
from typing import Tuple, Optional
import warnings
from scipy.interpolate import RegularGridInterpolator
import torch

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

    def _get_wind_at_points(self, lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lons = lons % 360
        pos = np.stack([lats, lons], axis=-1)
        return (
            self.u_interp(pos),
            self.v_interp(pos)
        )
    
    
    def generate_trajectory(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trajectories starting from given lat/lon arrays.
        lat: (batch_size, 1)
        lon: (batch_size, 1)
        
        Uses RK4 integration for better accuracy.
        Return only start and end points
        
        Args:
            lat: Starting latitude array
            lon: Starting longitude array
            
        Returns:
            Tuple of (start_points, end_points) where each is a batch of (lat,lon) pairs
        """
        # Cast inputs to float numpy arrays
        lat = np.array(lat, dtype=np.float32)
        lon = np.array(lon, dtype=np.float32)
        
        batch_size = len(lat)
        
        # Initialize trajectory
        current_lats = lat
        current_lons = lon % 360
        
        # Integrate forward
        for _ in range(self.n_steps):
            dlats, dlons = self._rk4_step(current_lats, current_lons)
            current_lats += dlats
            current_lons += dlons
            current_lats, current_lons = self._wrap_lat_lon(current_lats, current_lons)
            
        start_points = np.stack([lat, lon % 360], axis=-1) # (batch_size, 2)
        end_points = np.stack([current_lats, current_lons], axis=-1) # (batch_size, 2)

        # Normalize trajectory to be between 0 and 1. Lats are between -90 and 90, lons are between 0 and 360
        start_points[:, 0] = (start_points[:, 0] + 90) / 180
        start_points[:, 1] = (start_points[:, 1]) / 360
        end_points[:, 0] = (end_points[:, 0] + 90) / 180
        end_points[:, 1] = (end_points[:, 1]) / 360
        
        return start_points, end_points

    def _wrap_lat_lon(self, lats, lons):
        '''
        Note that this is completely incorrect and does not represent the real world. 

        '''
        # Clip longitude to be within 0 and 360
        lons = np.clip(lons, 0, 360)
        # Clip latitude to be within -90 and 90
        lats = np.clip(lats, -90, 90)
        return lats, lons


        # # Wrap latitude around the globe
        # if lat > 90:
        #     # lon = 360-lon
        #     lat = 180 - lat
        # elif lat < -90:
        #     # lon = 360-lon
        #     lat = -180 - lat
        # return lat, lon
    
    def generate_full_trajectory(self, start_lats: np.ndarray, start_lons: np.ndarray) -> np.ndarray:
        """
        Generate a full trajectory including intermediate points.
        
        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
        
        Returns:
            trajectory: List of (lat, lon) pairs for entire trajectory
        """

        # Cast start_lats and start_lons to float numpy arrays
        start_lats = np.array(start_lats, dtype=np.float32)
        start_lons = np.array(start_lons, dtype=np.float32)

        batch_size = len(start_lats)

        trajectory = np.zeros((batch_size, self.n_steps + 1, 2))
        trajectory[:, 0, :] = np.stack([start_lats, start_lons % 360], axis=-1)
        current_lats = start_lats
        current_lons = start_lons % 360
        
        for t in range(self.n_steps):
            dlats, dlons = self._rk4_step(current_lats, current_lons)
            current_lats += dlats
            current_lons += dlons
            current_lats, current_lons = self._wrap_lat_lon(current_lats, current_lons)
            trajectory[:, t+1, :] = np.stack([current_lats, current_lons], axis=-1)
        
        # Normalize trajectory to be between -1 and 1. Lats are between -90 and 90, lons are between 0 and 360
        trajectory[:, :, 0] = (trajectory[:, :, 0] + 90) / 180
        trajectory[:, :, 1] = (trajectory[:, :, 1] + 180) / 360
        return np.array(trajectory)
    
    def _rk4_step(self, lats, lons):
        u1, v1 = self._get_wind_at_points(lats, lons)
        # These are derivatives, not steps --> remember to multiply by dt
        k1_lats = v1
        k1_lons = u1/np.cos(np.radians(lats))

        u2, v2 = self._get_wind_at_points(lats + self.dt * k1_lats / 2 , lons + self.dt * k1_lons / 2)
        k2_lats = v2
        k2_lons = u2/np.cos(np.radians(lats + self.dt *  v1 / 2))

        u3, v3 = self._get_wind_at_points(lats + self.dt * k2_lats / 2 , lons + self.dt * k2_lons / 2)
        k3_lats = v3
        k3_lons = u3/np.cos(np.radians(lats + self.dt * k2_lats / 2))

        u4, v4 = self._get_wind_at_points(lats + self.dt * k3_lats, lons + self.dt * k3_lons)
        k4_lats = v4
        k4_lons = u4/np.cos(np.radians(lats + self.dt * k3_lats))

        dlats = self.dt * (k1_lats + 2*k2_lats + 2*k3_lats + k4_lats) / 6
        dlons = self.dt * (k1_lons + 2*k2_lons + 2*k3_lons + k4_lons) / 6

        return dlats, dlons

    def generate_batch(self, batch_size: int, full:bool=False):
        start_lats = np.arcsin(np.random.uniform(-1, 1, size=batch_size)) * (180/np.pi)
        start_lons = np.random.uniform(0, 360, size=batch_size)
        
        if full:
            return self.generate_full_trajectory(start_lats, start_lons)
        else:
            return self.generate_trajectory(start_lats, start_lons)

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