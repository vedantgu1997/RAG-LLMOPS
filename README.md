# Architecture Overview

## 1. Frontend Layer

### Web UI
- **Technology**: HTML/CSS/JavaScript templates  
- **Framework**: Jinja2 templates served by FastAPI  
- **Purpose**: User interface for document upload and interaction  

### API Gateway
- **Framework**: FastAPI  
- **Port**: 8000  
- **Features**: CORS enabled, static file serving, template rendering  

---

## 2. API Endpoints

| Endpoint   | Method | Purpose                        |
|------------|--------|--------------------------------|
| `/`        | GET    | Main UI interface              |
| `/health`  | GET    | System health check            |
| `/analyze` | POST   | Single document analysis       |
| `/compare` | POST   | Document comparison            |
| `/chat`    | POST   | Conversational RAG interface   |

---

## 3. Core Services

### Document Handler
- **File**: `data_ingestion.py`  
- **Purpose**: PDF processing and file management  
- **Features**: File upload, text extraction, document storage  

### Document Analyzer
- **File**: `data_analysis.py`  
- **Purpose**: LLM-powered document analysis  
- **Features**: Content analysis, summarization, key insights extraction  

### Document Comparator
- **File**: `data_comparator.py`  
- **Purpose**: Side-by-side document comparison  
- **Features**: LLM-based comparison, structured output as DataFrame  

### Conversational RAG
- **File**: `retrieval.py`  
- **Purpose**: Chat interface with document context  
- **Features**: Question answering, conversation history, context retrieval  

### Chat Ingestor
- **File**: `data_ingestion.py`  
- **Purpose**: Document ingestion for RAG system  
- **Features**: Document chunking, embedding generation, vector storage  

---

## 4. Utilities & Infrastructure

### Model Loader
- **File**: `model_loader.py`  
- **Purpose**: Centralized LLM and embedding model management  
- **Features**:
  - Multi-provider support (OpenAI, Groq)  
  - Environment validation  
  - Model configuration management  

### Config Loader
- **File**: `config_loader.py`  
- **Purpose**: YAML configuration management  
- **Features**: Configuration loading, validation, defaults  

### Custom Logger
- **File**: `custom_logger.py`  
- **Purpose**: Structured JSON logging  
- **Features**:
  - Console and file logging  
  - Timestamped log files  
  - Structured output with metadata  

### Custom Exception
- **File**: `custom_exception.py`  
- **Purpose**: Application-specific error handling  
- **Features**: Standardized error responses, detailed error context  

---

## 5. AI/ML Components

### OpenAI Integration
- **Models**: GPT-3.5-turbo, GPT-4  
- **Purpose**: Text generation, analysis, comparison  
- **API**: OpenAI API via LangChain  

### Groq Integration
- **Models**: Mixtral, LLaMA variants  
- **Purpose**: Fast inference alternative to OpenAI  
- **API**: Groq API via LangChain  

### OpenAI Embeddings
- **Model**: `text-embedding-ada-002`  
- **Purpose**: Document vectorization for semantic search  
- **Usage**: RAG retrieval, similarity matching  

### LangChain Framework
- **Purpose**: Chain orchestration and prompt management  
- **Features**:
  - LCEL (LangChain Expression Language) chains  
  - Prompt templates  
  - Output parsers  

---

## 6. Data Layer

### FAISS Vector Database
- **Purpose**: Store and retrieve document embeddings  
- **Features**:
  - Fast similarity search  
  - Scalable vector storage  
  - Index persistence  

### File Storage
- **Purpose**: Store uploaded documents  
- **Structure**: Organized by upload sessions  
- **Cleanup**: Automatic old session cleanup  

### Log Files
- **Purpose**: Application logging and monitoring  
- **Format**: JSON structured logs  
- **Rotation**: Timestamped files per session  

### Configuration Files
- **File**: `config.yaml`  
- **Purpose**: Application configuration  
- **Contents**: Model settings, database configs, API parameters  

---

## 7. External Dependencies

### OpenAI API
- **Services**: GPT models, embeddings  
- **Authentication**: API key via environment variables  
- **Usage**: Text generation, document analysis  

### Groq API
- **Services**: Fast LLM inference  
- **Authentication**: API key via environment variables  
- **Usage**: Alternative to OpenAI for speed optimization  

### Environment Variables
- **Required**:
  - `OPENAI_API_KEY`  
  - `GROQ_API_KEY`  
- **Optional**:
  - `LLM_PROVIDER`  
  - `FAISS_BASE`  
  - `UPLOAD_BASE`  


---

## Key Features

### Multi-Provider LLM Support
- Seamless switching between OpenAI and Groq  
- Environment-based provider selection  
- Consistent interface across providers  

### Robust Error Handling
- Custom exception classes  
- Detailed error logging  
- User-friendly error messages  

### Scalable Architecture
- Modular component design  
- Clear separation of concerns  
- Easy to extend and maintain  

### Comprehensive Logging
- Structured JSON logs  
- Multiple log levels  
- File and console output  

### Configuration Management
- YAML-based configuration  
- Environment variable override  
- Validation and defaults  

---

**This architecture provides a solid foundation for a production-ready RAG application with document processing, analysis, comparison, and conversational capabilities.**
