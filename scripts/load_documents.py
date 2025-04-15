from pathlib import Path
import fitz  # PyMuPDF

def load_pdf_text(file_path):
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "text": text,
            "page": i + 1,
            "source": Path(file_path).name
        })
    return pages

def load_all_documents(data_dir="data"):
    documents = []
    data_path = Path(data_dir)
    for file in data_path.glob("*.pdf"):
        documents.extend(load_pdf_text(file))
    return documents

if __name__ == "__main__":
    docs = load_all_documents()
    print(f"Loaded {len(docs)} pages from documents.")
