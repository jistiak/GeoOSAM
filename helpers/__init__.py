"""
Geo-OSAM Detection Helpers

Clean architecture for class-specific object detection logic.
Each helper contains proven, isolated logic for different object types.
"""

from .base_helper import BaseDetectionHelper
from .vegetation_helper import VegetationHelper
from .residential_helper import ResidentialHelper
from .vehicle_helper import VehicleHelper
from .buildings_helper import BuildingsHelper
from .water_helper import WaterHelper
from .agriculture_helper import AgricultureHelper
from .road_helper import RoadHelper
from .general_helper import GeneralHelper
from .vessels_helper import VesselsHelper

CLASS_FAMILIES = {
    'vegetation': {
        'vegetation', 'grass', 'greenfield', 'tree canopy', 'artificial turf'
    },
    'building': {
        'buildings', 'building', 'industrial', 'glass roof',
        'green roof', 'red roof', 'dark roof', 'industrial roof', 'pv',
        'thermo', 'window', 'solar tube'
    },
    'residential': {
        'residential'
    },
    'vehicle': {
        'vehicle', 'cars'
    },
    'vessel': {
        'vessels', 'vessel', 'ship', 'ships', 'boat', 'boats'
    },
    'water': {
        'water'
    },
    'agriculture': {
        'agriculture', 'field'
    },
    'road': {
        'road', 'roads', 'railway', 'bike lane'
    },
}


def normalize_class_name(class_name):
    """Normalize class names for alias matching."""
    return " ".join((class_name or "").strip().casefold().split())


def get_class_family(class_name):
    """Return the helper family for a class name."""
    normalized = normalize_class_name(class_name)
    for family, class_names in CLASS_FAMILIES.items():
        if normalized in class_names:
            return family
    return 'general'


def class_uses_helper(class_name, *families):
    """Check whether a class belongs to one of the requested helper families."""
    return get_class_family(class_name) in families


def create_detection_helper(class_name, min_object_size=50, max_objects=25):
    """Factory to create appropriate helper for each class"""
    family = get_class_family(class_name)

    if family == 'vegetation':
        return VegetationHelper(class_name, min_object_size, max_objects)
    elif family == 'building':
        return BuildingsHelper(class_name, min_object_size, max_objects)
    elif family == 'residential':
        return ResidentialHelper(class_name, min_object_size, max_objects)
    elif family == 'vehicle':
        return VehicleHelper(class_name, min_object_size, max_objects)
    elif family == 'vessel':
        return VesselsHelper(class_name, min_object_size, max_objects)
    elif family == 'water':
        return WaterHelper(class_name, min_object_size, max_objects)
    elif family == 'agriculture':
        return AgricultureHelper(class_name, min_object_size, max_objects)
    elif family == 'road':
        return RoadHelper(class_name, min_object_size, max_objects)
    else:
        # Default helper for other classes - use general helper as fallback
        return GeneralHelper(class_name, min_object_size, max_objects)

__all__ = [
    'BaseDetectionHelper',
    'VegetationHelper', 
    'ResidentialHelper',
    'VehicleHelper',
    'BuildingsHelper',
    'WaterHelper',
    'AgricultureHelper',
    'RoadHelper',
    'GeneralHelper',
    'VesselsHelper',
    'CLASS_FAMILIES',
    'normalize_class_name',
    'get_class_family',
    'class_uses_helper',
    'create_detection_helper'
]
