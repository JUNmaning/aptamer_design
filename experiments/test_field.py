from fields.electrostatic import compute_electrostatic_field

field, grid = compute_electrostatic_field(
    "data/together.pdb"
)

print("Field shape:", field.shape)
print("Field min/max:", field.min(), field.max())