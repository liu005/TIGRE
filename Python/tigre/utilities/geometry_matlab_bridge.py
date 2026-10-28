"""MAT/YAML/NPZ interchange for the MATLAB GUI; run with the recon interpreter."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.io import loadmat, savemat

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tigre.utilities.geometry_io import load_geometry_file, save_geometry_yaml
from tigre.utilities.detector_geometry import DetectorSurfaceGeometry


def to_mat(source, target):
    document = load_geometry_file(source)
    surface = document.surface
    arrays = dict(source_positions_mm=surface.source_positions_mm.T,
                  pixel_centers_mm=surface.pixel_centers_mm.transpose(1, 2, 3, 0),
                  valid_mask=surface.valid_mask.transpose(1, 2, 0))
    if surface._module_slices:
        arrays['module_slices'] = np.array([[r.start, r.stop, c.start, c.stop]
                                          for r, c in surface._module_slices])
    result = {'surface': arrays, 'geo': np.empty((0, 0))}
    if document.settings is not None:
        result['settings'] = {k: np.empty((0, 0)) if v is None else v
                              for k, v in document.settings.items()}
    if document.geometry is not None:
        result['geo'] = dict(sVoxel=np.asarray(document.geometry.sVoxel)[::-1],
                             offOrigin=np.atleast_2d(document.geometry.offOrigin)[:, ::-1].T)
        if hasattr(document.geometry, 'nVoxel'):
            result['geo']['nVoxel'] = document.geometry.nVoxel[::-1]
            result['geo']['dVoxel'] = document.geometry.dVoxel[::-1]
        for name in ('acquisition', 'motion_mode', 'view_times_s'):
            if hasattr(document.geometry, name):
                result['geo'][name] = getattr(document.geometry, name)
    savemat(target, result)


def from_mat(source, target):
    document = loadmat(source, simplify_cells=True)
    geo = document.get('geo')
    volume = None
    if isinstance(geo, dict):
        from types import SimpleNamespace
        offsets = np.asarray(geo['offOrigin'])
        offsets = offsets[::-1] if offsets.ndim == 1 else offsets.T[:, ::-1]
        volume = SimpleNamespace(sVoxel=np.asarray(geo['sVoxel'])[::-1],
                                 offOrigin=offsets)
        if 'nVoxel' in geo:
            volume.nVoxel = np.asarray(geo['nVoxel'])[::-1]
        for name in ('acquisition', 'motion_mode', 'view_times_s'):
            if name in geo:
                setattr(volume, name, geo[name])
    if isinstance(document.get('settings'), dict):
        settings = document['settings']
        for key in ('transverse_centers_mm', 'normal_offsets_mm', 'z_offsets_mm',
                    'panel_angles_deg', 'panel_x_angles_deg', 'panel_y_angles_deg'):
            settings[key] = np.atleast_1d(settings[key])
        if not isinstance(settings.get('acquisition'), dict):
            settings['acquisition'] = None
        for key in ('n_views', 'n_modules', 'column_gap'):
            value = float(settings[key])
            if not value.is_integer():
                raise ValueError(f'{key} must be an integer')
            settings[key] = int(value)
        shape = np.asarray(settings['module_shape'])
        if not np.equal(shape, np.floor(shape)).all():
            raise ValueError('module_shape must contain integers')
        settings['module_shape'] = shape.astype(int)
        if np.asarray(settings.get('source_start_mm', [])).size == 0:
            settings['source_start_mm'] = None
        save_geometry_yaml(target, settings=settings, geometry=volume)
    else:
        # Read unsqueezed arrays: singleton rows/views carry physical meaning.
        raw = loadmat(source, struct_as_record=False, squeeze_me=False)['surface'][0, 0]
        pixels = raw.pixel_centers_mm
        if pixels.ndim == 3:
            pixels = pixels[..., None]
        mask = raw.valid_mask
        if mask.ndim == 2:
            mask = mask[..., None]
        modules = getattr(raw, 'module_slices', None)
        if modules is not None:
            modules = [(slice(int(a), int(b)), slice(int(c), int(d))) for a, b, c, d in modules]
        surface = DetectorSurfaceGeometry(raw.source_positions_mm.T,
            pixels.transpose(3, 0, 1, 2), mask.transpose(2, 0, 1), module_slices=modules)
        save_geometry_yaml(target, surface=surface, geometry=volume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['to-mat', 'from-mat'])
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()
    (to_mat if args.operation == 'to-mat' else from_mat)(args.source, args.target)
