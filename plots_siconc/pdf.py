import os, pathlib, glob
from fpdf import FPDF               # make sure `pip install fpdf`

def make_gallery_pdf(
        folder=".",                 # ← default: current dir
        pdf_name="gallery.pdf",
        ncols=2, nrows=3,
        recursive=True):
    """
    Collect every *.png* in *folder* (or sub-folders) and lay them out
    as small-multiples on A4 pages – then save a single PDF.

    * All paths are handled ABSOLUTELY → no “plots_emidms/plots_emidms”
      duplication possible.
    * Works even when you call the function from a notebook that already
      sits inside the image folder.
    """

    # 1. resolve once – ABSOLUTE path, no chdir any more
    folder = pathlib.Path(folder).expanduser().resolve()

    # 2. collect PNGs (recursive optional)
    pattern = "**/*.png" if recursive else "*.png"
    pngs    = sorted(folder.glob(pattern))
    if not pngs:
        raise FileNotFoundError(f"No PNG files under {folder}")

    # 3. start the PDF
    pdf = FPDF(unit="pt", format="A4")
    pw, ph = pdf.w, pdf.h
    margin = 40                                       # pt
    cell_w = (pw - 2*margin) / ncols
    cell_h = (ph - 2*margin) / nrows

    # 4. lay out pages
    for i, png in enumerate(pngs):
        if i % (ncols*nrows) == 0:
            pdf.add_page()
        col = (i % ncols)
        row = (i // ncols) % nrows
        x   = margin + col * cell_w
        y   = margin + row * cell_h
        pdf.image(str(png), x=x, y=y, w=cell_w, h=cell_h)

    # 5. write file in the *same* folder as the notebook
    out_path = folder / pdf_name
    pdf.output(str(out_path))
    print(f"✅  Saved {out_path.relative_to(pathlib.Path.cwd())}")