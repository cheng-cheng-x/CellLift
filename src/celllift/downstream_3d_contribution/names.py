"""Display names for structural measurements and attribution groups."""
NAMES = {'nucleus_log_volume': 'Nuclear volume (natural logarithm)', 'cell_log_volume': 'Cell volume (natural logarithm)', 'nucleus_log_axis12': 'Nuclear axis 1/2 ratio (logarithm)', 'nucleus_log_axis23': 'Nuclear axis 2/3 ratio (logarithm)', 'cell_log_axis12': 'Cell axis 1/2 ratio (logarithm)', 'cell_log_axis23': 'Cell axis 2/3 ratio (logarithm)', 'nucleus_z_orientation': 'Nuclear axial orientation', 'cell_z_orientation': 'Cell axial orientation', 'log_nucleus_cytoplasm_ratio': 'Nuclear-to-cytoplasmic volume ratio (logarithm)', 'offset_xy_relative': 'Relative nuclear-cell in-plane offset', 'offset_z_relative': 'Relative nuclear-cell axial offset', 'offset_xyz_relative': 'Relative nuclear-cell spatial offset', 'relative_axial_gap': 'Relative axial gap between neighbouring nuclei', 'nucleus_direction_spacing': 'Relative directional spacing between nuclei', 'cell_direction_spacing': 'Relative directional spacing between cells', 'spacing_3d_minus_2d': 'Three-dimensional minus two-dimensional directional spacing', 'neighbor_abs_nucleus_log_volume': 'Absolute neighbour difference in log nuclear volume', 'neighbor_abs_cell_log_volume': 'Absolute neighbour difference in log cell volume', 'neighbor_abs_nucleus_axis': 'Absolute neighbour difference in nuclear axis ratio', 'neighbor_abs_cell_axis': 'Absolute neighbour difference in cell axis ratio', 'descriptor_block': 'Three-dimensional descriptor group', 'nucleus_matrix_path': 'Nuclear shape matrix', 'cell_matrix_path': 'Cell shape matrix', 'cell_offset_path': 'Nuclear-cell offset', 'axial_position_path': 'Axial-position-derived input'}

def title(feature):
    if '__' not in feature:
        return NAMES.get(feature, feature)
    base, stat = feature.split('__')
    return NAMES.get(base, base) + (' (median)' if stat == 'median' else ' (IQR)')

def interpretation(feature):
    base, stat = feature.split('__')
    if 'direction_spacing' in base or base == 'spacing_3d_minus_2d':
        meaning = 'Directional spacing accounts for object shape and connecting direction and is normalized by object size. '
    elif 'neighbor_abs' in base:
        meaning = 'Neighbour differences quantify local dissimilarity; volume differences use logarithmic values. '
    elif 'orientation' in base:
        meaning = 'Larger values generally indicate stronger axial alignment of the long axis, subject to the nearly isotropic-object convention. '
    elif 'offset' in base:
        meaning = 'Nuclear-cell displacement is normalized by the equivalent nuclear radius. '
    elif 'cytoplasm' in base:
        meaning = 'Cytoplasmic volume is cell volume minus nuclear volume. '
    elif 'volume' in base:
        meaning = 'Object volume is represented on the natural-logarithm scale. '
    else:
        meaning = 'The measurement describes inferred object shape or spatial position. '
    return meaning + ('IQR measures within-unit dispersion; patient summaries follow patch-to-slide-to-patient aggregation.' if stat == 'iqr' else 'The median describes the typical within-unit value.')
