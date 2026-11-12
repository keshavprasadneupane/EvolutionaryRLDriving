"""
Top-down racing engine (Box2D 3.2). Raw simulation, no env logic.
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['CarConfig', 'CarObservation', 'CarSpec', 'DriveMode', 'LidarConfig', 'MultiCarWorld', 'RaceWorld', 'SingleCarWorld', 'SurfaceType', 'TDC_DOWN', 'TDC_LEFT', 'TDC_RIGHT', 'TDC_UP', 'TD_CAT_CAR', 'TD_CAT_WALL', 'TrackConfig', 'WorldConfig']
class DriveMode:
    """
    Members:
    
      Discrete
    
      Continuous
    """
    Continuous: typing.ClassVar[DriveMode]
    Discrete: typing.ClassVar[DriveMode]
    __members__: typing.ClassVar[dict[str, DriveMode]]
    @typing.overload
    def __eq__(self, other: DriveMode) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: DriveMode) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SurfaceType:
    """
    Members:
    
      Track
    
      Grass
    
      Wall
    
      Outside
    """
    Grass: typing.ClassVar[SurfaceType]
    Outside: typing.ClassVar[SurfaceType]
    Track: typing.ClassVar[SurfaceType]
    Wall: typing.ClassVar[SurfaceType]
    __members__: typing.ClassVar[dict[str, SurfaceType]]
    @typing.overload
    def __eq__(self, other: SurfaceType) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: SurfaceType) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TrackConfig:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: TrackConfig) -> None:
        ...
    @property
    def control_points(self) -> int:
        ...
    @control_points.setter
    def control_points(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def geom_scale(self) -> float:
        ...
    @geom_scale.setter
    def geom_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def grass_padding(self) -> float:
        ...
    @grass_padding.setter
    def grass_padding(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def grass_width(self) -> float:
        ...
    @grass_width.setter
    def grass_width(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def track_half_width(self) -> float:
        ...
    @track_half_width.setter
    def track_half_width(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def turn_rate(self) -> float:
        ...
    @turn_rate.setter
    def turn_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wall_thickness(self) -> float:
        ...
    @wall_thickness.setter
    def wall_thickness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class WorldConfig:
    cars_collide: bool
    track: TrackConfig
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: WorldConfig) -> None:
        ...
    @property
    def grass_traction(self) -> float:
        ...
    @grass_traction.setter
    def grass_traction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def grid_column_spacing(self) -> float:
        ...
    @grid_column_spacing.setter
    def grid_column_spacing(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def grid_columns(self) -> int:
        ...
    @grid_columns.setter
    def grid_columns(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def grid_row_spacing_samples(self) -> int:
        ...
    @grid_row_spacing_samples.setter
    def grid_row_spacing_samples(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_track_attempts(self) -> int:
        ...
    @max_track_attempts.setter
    def max_track_attempts(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def num_checkpoints(self) -> int:
        ...
    @num_checkpoints.setter
    def num_checkpoints(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def sub_step_count(self) -> int:
        ...
    @sub_step_count.setter
    def sub_step_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def time_step(self) -> float:
        ...
    @time_step.setter
    def time_step(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def track_traction(self) -> float:
        ...
    @track_traction.setter
    def track_traction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class LidarConfig:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: LidarConfig) -> None:
        ...
    @property
    def fov(self) -> float:
        ...
    @fov.setter
    def fov(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hit_mask(self) -> int:
        ...
    @hit_mask.setter
    def hit_mask(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def local_origin(self) -> tuple[float, float]:
        ...
    @local_origin.setter
    def local_origin(self, arg1: tuple[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def look_angle(self) -> float:
        ...
    @look_angle.setter
    def look_angle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def range(self) -> float:
        ...
    @range.setter
    def range(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ray_count(self) -> int:
        ...
    @ray_count.setter
    def ray_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class CarConfig:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: CarConfig) -> None:
        ...
    def center_y(self) -> float:
        ...
    @property
    def angular_damping(self) -> float:
        ...
    @angular_damping.setter
    def angular_damping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def angular_friction(self) -> float:
        ...
    @angular_friction.setter
    def angular_friction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def body_density(self) -> float:
        ...
    @body_density.setter
    def body_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def body_vertices(self) -> list[tuple[float, float]]:
        ...
    @body_vertices.setter
    def body_vertices(self, arg1: collections.abc.Sequence[tuple[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex]]) -> None:
        ...
    @property
    def forward_drag(self) -> float:
        ...
    @forward_drag.setter
    def forward_drag(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_backward_speed(self) -> float:
        ...
    @max_backward_speed.setter
    def max_backward_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_forward_speed(self) -> float:
        ...
    @max_forward_speed.setter
    def max_forward_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def steer_lock_angle(self) -> float:
        ...
    @steer_lock_angle.setter
    def steer_lock_angle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def steer_speed(self) -> float:
        ...
    @steer_speed.setter
    def steer_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CarSpec:
    car: CarConfig
    lidar: LidarConfig
    mode: DriveMode
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: CarSpec) -> None:
        ...
class CarObservation:
    @property
    def angle(self) -> float:
        ...
    @property
    def angular_velocity(self) -> float:
        ...
    @property
    def checkpoints_crossed(self) -> int:
        ...
    @property
    def correct_way(self) -> bool:
        ...
    @property
    def heading_alignment(self) -> float:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def laps_completed(self) -> int:
        ...
    @property
    def lateral_offset(self) -> float:
        ...
    @property
    def local_velocity(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        x = sideways (right +), y = forward
        """
    @property
    def max_checkpoints(self) -> int:
        ...
    @property
    def position(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        body origin (the rear of the car) in world units
        """
    @property
    def ray_surfaces(self) -> numpy.typing.NDArray[numpy.uint8]:
        ...
    @property
    def rays(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        lidar hit fraction per ray, 1 = nothing in range
        """
    @property
    def speed(self) -> float:
        ...
    @property
    def steer_angle(self) -> float:
        ...
    @property
    def surface(self) -> SurfaceType:
        ...
    @property
    def track_index(self) -> int:
        ...
    @property
    def traction(self) -> float:
        ...
    @property
    def world_velocity(self) -> numpy.typing.NDArray[numpy.float32]:
        ...
class RaceWorld:
    def get_static(self) -> dict:
        """
        track / grass / walls / checkpoints as numpy arrays
        """
    def reset(self, new_track: bool = False, seed: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        put every car back on its slot; new_track=True builds a new map (call get_static() again). seed reseeds the track generator, 0 = random. Must be called before the first step().
        """
    def step(self, dt: typing.SupportsFloat | typing.SupportsIndex = 0.0, sub_step_count: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        advance the simulation; raises RuntimeError if reset() has not been called yet
        """
    @property
    def car_count(self) -> int:
        ...
    @property
    def config(self) -> WorldConfig:
        """
        a copy of the WorldConfig this world was built with
        """
    @property
    def needs_reset(self) -> bool:
        """
        True until reset() has been called (step() would raise)
        """
class SingleCarWorld(RaceWorld):
    def __init__(self, world_config: WorldConfig = ..., car_config: CarConfig = ..., lidar_config: LidarConfig = ..., mode: DriveMode = ...) -> None:
        ...
    def cast_view(self) -> dict:
        ...
    def current_control(self) -> tuple:
        ...
    def drive_mode(self) -> DriveMode:
        ...
    def get_observation(self) -> CarObservation:
        ...
    def reset_car(self) -> None:
        ...
    def set_continuous_input(self, arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_discrete_input(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_drive_mode(self, arg0: DriveMode) -> None:
        ...
    def toggle_drive_mode(self) -> None:
        ...
class MultiCarWorld(RaceWorld):
    def __init__(self, world_config: WorldConfig = ...) -> None:
        ...
    def add_car(self, spec: CarSpec = ...) -> int:
        ...
    def cast_view(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> dict:
        ...
    def current_control(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> tuple:
        ...
    def drive_mode(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> DriveMode:
        ...
    def get_observation(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> CarObservation:
        ...
    def get_observations(self) -> list[CarObservation]:
        """
        every car, in the order of ids()
        """
    def has_car(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def ids(self) -> list[int]:
        ...
    def remove_car(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def reset_car(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_continuous_input(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_continuous_inputs(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        ...
    def set_discrete_input(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_discrete_inputs(self, arg0: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], arg1: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
    def set_drive_mode(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: DriveMode) -> None:
        ...
TDC_DOWN: int = 8
TDC_LEFT: int = 1
TDC_RIGHT: int = 2
TDC_UP: int = 4
TD_CAT_CAR: int = 2
TD_CAT_WALL: int = 4
