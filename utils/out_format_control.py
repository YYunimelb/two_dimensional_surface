def print_surface_summary(surface_markers):
    """
    Print the positions (1-based index) of atoms on the top surface (1), bottom surface (-1), and isolated (3).

    Parameters:
    - surface_markers (list): List of surface markers where 1 = top surface, -1 = bottom surface, 3 = isolated atom.
    """
    top_surface_indices = [i + 1 for i, marker in enumerate(surface_markers) if marker == 1]
    bottom_surface_indices = [i + 1 for i, marker in enumerate(surface_markers) if marker == -1]
    isolated_indices = [i + 1 for i, marker in enumerate(surface_markers) if marker == 3]

    print("Top surface atoms (1):", top_surface_indices)
    print("Bottom surface atoms (-1):", bottom_surface_indices)
    print("Isolated atoms (3):", isolated_indices)


def format_surface_atoms_output(surface_atoms):
    """
    Format the surface atoms output for better readability.

    Parameters:
    - surface_atoms (dict): Dictionary with 'top_surface' and 'bottom_surface' containing atom details.

    Returns:
    - str: Formatted string with structured output.
    """
    output = ""

    def format_atom_info(atom_info):
        info = f"Atom Index: {atom_info['index']}, Element: {atom_info['element']}, Bond Count: {atom_info['bond_count']}\n"
        info += "  Bonded Atoms:\n"
        for bonded in atom_info['bonded_atoms']:
            info += (
                f"    - Bonded Atom Index: {bonded['bonded_atom_index']}, "
                f"Type: {bonded['bonded_atom_type']}, "
                f"Distance: {bonded['bond_distance']:.3f}\n"
            )
        return info

    # Format top surface
    output += "Top Surface Atoms:\n"
    for atom in surface_atoms['top_surface']:
        output += format_atom_info(atom)

    # Format bottom surface
    output += "\nBottom Surface Atoms:\n"
    for atom in surface_atoms['bottom_surface']:
        output += format_atom_info(atom)

    return output

