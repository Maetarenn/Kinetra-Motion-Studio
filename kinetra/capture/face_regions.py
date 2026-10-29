"""Which face-mesh vertices belong to the eyes, brows, nose and lips (the regions that carry a performance)."""
from ..character import MeshBuilder
from .models import face_connections

# approximate nose set of the 468-point mesh: bridge, tip, septum and nostril wings
NOSE = (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 129, 358, 49, 279, 48, 278, 64, 294, 115, 344,
        220, 440, 45, 275, 219, 439, 209, 429, 198, 420, 131, 360)


def _ids(name):
    return sorted({i for a, b in face_connections(name) for i in (a, b) if i < 468})


def feature_regions():
    return {"lips": _ids("LIPS"), "eyes": sorted(set(_ids("LEFT_EYE")) | set(_ids("RIGHT_EYE"))),
            "brows": sorted(set(_ids("LEFT_EYEBROW")) | set(_ids("RIGHT_EYEBROW"))), "nose": sorted(NOSE)}


def feature_triangles(regions=None):
    """Mesh triangles whose three vertices all lie in one feature region."""
    regions = regions or feature_regions()
    mb = MeshBuilder()
    sets = [set(v) for v in regions.values()]
    return [tuple(int(i) for i in t) for t in mb.tris if any(all(int(i) in s for i in t) for s in sets)]
