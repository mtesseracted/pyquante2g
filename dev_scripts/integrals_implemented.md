# Integrals Implemented

Snapshot transcribed from the OpenOffice spreadsheet.

Legend:
- `implemented`: green in the source sheet
- `planned`: orange in the source sheet
- blank: no status marked in the source sheet

| Method | Arbitrary Precision | Scalar NumPy | ArrayOps |
| --- | --- | --- | --- |
| `THO66` | Overlap, Kinetic, NAI, ERI (`implemented`) |  | Overlap, Kinetic, NAI (`implemented`, note: `move to cart_gto`), ERI (`implemented`) |
| `HT95` |  | NAI (`planned`) |  |
| `HH12` |  | Overlap, Kinetic (`implemented`) |  |

Expanded matrix:

| Method | AP Overlap | AP Kinetic | AP NAI | AP ERI | NumPy Overlap | NumPy Kinetic | NumPy NAI | NumPy ERI | ArrayOps Overlap | ArrayOps Kinetic | ArrayOps NAI | ArrayOps ERI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `THO66` | implemented | implemented | implemented | implemented |  |  |  |  | implemented | implemented | implemented (`move to cart_gto`) | implemented |
| `HT95` |  |  |  |  |  |  | planned |  |  |  |  |  |
| `HH12` |  |  |  |  | implemented | implemented |  |  |  |  |  |  |
