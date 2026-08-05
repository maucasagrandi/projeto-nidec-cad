from src.utils.opencv_cad_compare import compare_cad_pages_opencv, export_comparison

result = compare_cad_pages_opencv(pdf1_bytes, pdf2_bytes)
export_comparison(result, output_dir="path/to/output")
