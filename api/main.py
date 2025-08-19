from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from src.doc_ingestion.data_ingestion import DocHandler, DocumentComparator, ChatIngestor
from src.doc_analyzer.data_analysis import DocumentAnalyzer
from src.doc_compare.data_comparator import DocumentComparatorLLM
from src.doc_chat.retrieval import ConversationalRAG

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

app = FastAPI(title='Document Portal API', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}

class FastAPIFileAdapter:
    """
    Adapt FastAPI UploadFile -> .name + .getbuffer() API
    """
    def __init__(self, uf: UploadFile) -> None:
        self._uf = uf
        self.name = uf.filename

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()

def _read_pdf_via_handler(handler: DocHandler, path: str) -> str:
    """
    Helper function to read PDF using the provided handler.
    """
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)
    if hasattr(handler, "read_"):
        return handler.read_(path) # type: ignore
    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")

@app.post("/analyze")
async def analyze_documents(file: UploadFile = File(...)) -> Any:
    try:
        dh = DocHandler()
        save_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = _read_pdf_via_handler(dh, save_path)
        analyzer = DocumentAnalyzer()
        analysis_result = analyzer.analyze_document(text)
        return JSONResponse(content=analysis_result)
    
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(FastAPIFileAdapter(reference), FastAPIFileAdapter(actual))
        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        return {"rows": df.to_dict(orient='records'), 'session_id': dc.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
    
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
    k: int = Form(5)
) -> Any:
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base = UPLOAD_BASE,
            faiss_base = FAISS_BASE,
            use_session_dirs = use_session_dirs,
            session_id = session_id or None
        )
        ci.build_retriever(wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k)
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
    
@app.post("/chat/query")
async def chat_query(
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5)
    ) -> Any:
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required when using session directories.")
        
        #Prepare fiass index path
        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"Index directory not found: {index_dir}")
        
        #Initialize LCEL-style RAG pipeline
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir)

        #optional for now we pass empty chat history
        response = rag.invoke(query, chat_history=[])

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    
# python -m uvicorn main:app --reload